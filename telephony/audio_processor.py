"""
Enhanced audio processing utilities for telephony integration with Google Cloud TTS.

Handles audio format conversion between Twilio and Voice AI Agent.
"""
import audioop
import numpy as np
import logging
import io
import wave
from typing import Tuple, Dict, Any
from scipy import signal

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio conversion between Twilio and Voice AI formats with improved telephony performance.
    Optimized for Google Cloud TTS integration.
    
    Twilio uses 8kHz mulaw encoding, while our Voice AI uses 16kHz PCM.
    """
    
    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM for Voice AI with enhanced noise filtering.
        
        Args:
            mulaw_data: Audio data in mulaw format
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Check if we have enough data to convert
            if len(mulaw_data) < 1000:
                logger.warning(f"Very small mulaw data: {len(mulaw_data)} bytes")
            
            # Convert mulaw to 16-bit PCM
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Resample from 8kHz to 16kHz
            pcm_data_16k, _ = audioop.ratecv(
                pcm_data, 2, 1, 
                SAMPLE_RATE_TWILIO, 
                SAMPLE_RATE_AI, 
                None
            )
            
            # Convert to numpy array (float32)
            audio_array = np.frombuffer(pcm_data_16k, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Apply enhanced audio filtering optimized for Deepgram
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(6, 100/(SAMPLE_RATE_AI/2), 'highpass')
            audio_array = signal.filtfilt(b, a, audio_array)
            
            # Apply band-pass filter for telephony freq range (300-3400 Hz)
            b, a = signal.butter(4, [300/(SAMPLE_RATE_AI/2), 3400/(SAMPLE_RATE_AI/2)], 'band')
            audio_array = signal.filtfilt(b, a, audio_array)
            
            # Apply a simple noise gate
            noise_threshold = 0.015  # Adjusted threshold
            audio_array = np.where(np.abs(audio_array) < noise_threshold, 0, audio_array)
            
            # Apply pre-emphasis filter to boost higher frequencies
            audio_array = np.append(audio_array[0], audio_array[1:] - 0.97 * audio_array[:-1])
            
            # Normalize for consistent volume
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array * (0.9 / max_val)
            
            # Check audio levels
            audio_level = np.mean(np.abs(audio_array)) * 100
            logger.debug(f"Converted {len(mulaw_data)} bytes to {len(audio_array)} samples. Audio level: {audio_level:.1f}%")
            
            # Apply a gain if audio is very quiet
            if audio_level < 1.0:  # Very quiet audio
                audio_array = audio_array * min(5.0, 5.0/audio_level)
                logger.debug(f"Applied gain to quiet audio. New level: {np.mean(np.abs(audio_array)) * 100:.1f}%")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            # Return an empty array rather than raising an exception
            return np.array([], dtype=np.float32)
    
    @staticmethod
    def pcm_to_mulaw(pcm_data: bytes, source_sample_rate: int = None) -> bytes:
        """
        Convert PCM audio from Google Cloud TTS to mulaw for Twilio with enhanced quality.
        
        Args:
            pcm_data: Audio data in PCM format
            source_sample_rate: Sample rate of the source PCM data (None to auto-detect)
            
        Returns:
            Audio data in mulaw format optimized for telephony
        """
        try:
            # Try to detect input format and parameters
            detected_sample_rate = source_sample_rate or SAMPLE_RATE_AI
            
            # If it looks like a WAV file, extract data from it
            if pcm_data[:4] == b'RIFF' and pcm_data[8:12] == b'WAVE':
                # Parse WAV header to get sample rate and extract raw PCM
                with io.BytesIO(pcm_data) as wav_io:
                    with wave.open(wav_io, 'rb') as wav_file:
                        detected_sample_rate = wav_file.getframerate()
                        n_channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        pcm_data = wav_file.readframes(wav_file.getnframes())
                        logger.debug(f"Detected WAV: {detected_sample_rate}Hz, {n_channels} channels, {sample_width} bytes/sample")
            
            # Ensure we have an even number of bytes for 16-bit samples
            if len(pcm_data) % 2 != 0:
                pcm_data = pcm_data + b'\x00'
                logger.debug("Padded audio data to make even length")
            
            # Amplify the signal a bit to ensure good volume over the phone
            try:
                pcm_data = audioop.mul(pcm_data, 2, 1.2)  # Amplify by 20%
            except Exception as amp_error:
                logger.warning(f"Could not amplify audio: {amp_error}")
            
            # Apply a slight compression to improve clarity over telephony
            try:
                # Compression: reduce dynamic range to make quieter sounds more audible
                pcm_data = audioop.compress(pcm_data, 2, 5, 2)  # parameters: fragment size=2, threshold=5, ratio=2:1
            except Exception as comp_error:
                logger.warning(f"Could not compress audio: {comp_error}")
            
            # Optimize frequency response for telephony
            # This can be done in numpy, but would require converting back and forth
            # Instead, we use the resampling step to handle this
            
            # Resample from detected rate to 8kHz for Twilio (this also acts as a low-pass filter)
            if detected_sample_rate != SAMPLE_RATE_TWILIO:
                try:
                    pcm_data_8k, _ = audioop.ratecv(
                        pcm_data, 2, 1, 
                        detected_sample_rate, 
                        SAMPLE_RATE_TWILIO, 
                        None
                    )
                    logger.debug(f"Resampled audio from {detected_sample_rate}Hz to {SAMPLE_RATE_TWILIO}Hz")
                except Exception as resample_error:
                    logger.error(f"Error resampling audio: {resample_error}")
                    pcm_data_8k = pcm_data  # Use original as fallback
            else:
                pcm_data_8k = pcm_data
            
            # Convert to mulaw - this gives us the 8-bit encoding Twilio expects
            mulaw_data = audioop.lin2ulaw(pcm_data_8k, 2)
            
            # Log the conversion details
            compression_ratio = len(pcm_data) / len(mulaw_data) if len(mulaw_data) > 0 else 0
            logger.info(f"Converted {len(pcm_data)} bytes of PCM to {len(mulaw_data)} bytes of mulaw (ratio: {compression_ratio:.1f}x)")
            
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error converting PCM to mulaw: {e}", exc_info=True)
            # Return empty data rather than raising an exception
            return b''
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    @staticmethod
    def enhance_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality by reducing noise and improving speech clarity.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Enhanced audio data
        """
        try:
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            # Telephone lines often have low frequency hum
            b, a = signal.butter(4, 80/(SAMPLE_RATE_AI/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply a mild de-emphasis filter to reduce hissing sounds in phone calls
            b, a = signal.butter(1, 3000/(SAMPLE_RATE_AI/2), 'low')
            de_emphasis = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Apply a simple noise gate to remove background noise
            noise_threshold = 0.005  # Adjust based on expected noise level
            noise_gate = np.where(np.abs(de_emphasis) < noise_threshold, 0, de_emphasis)
            
            # 4. Apply pre-emphasis filter to boost higher frequencies (for better speech detection)
            pre_emphasis = np.append(noise_gate[0], noise_gate[1:] - 0.97 * noise_gate[:-1])
            
            # 5. Normalize audio to have consistent volume
            if np.max(np.abs(pre_emphasis)) > 0:
                normalized = pre_emphasis / np.max(np.abs(pre_emphasis)) * 0.95
            else:
                normalized = pre_emphasis
            
            # 6. Apply a mild compression to even out volumes
            # Compression ratio 2:1 for values above threshold
            threshold = 0.2
            ratio = 0.5  # 2:1 compression
            
            def compressor(x, threshold, ratio):
                # If below threshold, leave it alone
                # If above threshold, compress it
                mask = np.abs(x) > threshold
                sign = np.sign(x)
                mag = np.abs(x)
                compressed = np.where(
                    mask,
                    threshold + (mag - threshold) * ratio,
                    mag
                )
                return sign * compressed
            
            compressed = compressor(normalized, threshold, ratio)
            
            # Re-normalize after compression
            if np.max(np.abs(compressed)) > 0:
                result = compressed / np.max(np.abs(compressed)) * 0.95
            else:
                result = compressed
                
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            # Return original audio if enhancement fails
            return audio_data
            
    @staticmethod
    def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Enhanced silence detection with frequency analysis.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Silence threshold
            
        Returns:
            True if audio is considered silence
        """
        try:
            # 1. Check energy level
            energy = np.mean(np.abs(audio_data))
            energy_silence = energy < threshold
            
            # Only do more expensive analysis if the energy check isn't conclusive
            if energy < threshold * 2:  # If energy is low but not definitely silent
                # 2. Check zero-crossing rate (white noise has high ZCR)
                zcr = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
                
                # 3. Check spectral flatness (noise typically has flatter spectrum)
                # Approximate with FFT magnitude variance
                fft_data = np.abs(np.fft.rfft(audio_data))
                spectral_flatness = np.std(fft_data) / (np.mean(fft_data) + 1e-10)
                
                # Combined decision - true silence has low energy, low-moderate ZCR, and low spectral flatness
                return energy_silence and zcr < 0.1 and spectral_flatness < 2.0
            
            # If energy is very low or very high, just use that criterion
            return energy_silence
            
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
            # Fall back to simple energy threshold
            return np.mean(np.abs(audio_data)) < threshold
    
    @staticmethod
    def get_audio_info(audio_data: bytes) -> Dict[str, Any]:
        """
        Get information about audio data.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Dictionary with audio information
        """
        info = {
            "size_bytes": len(audio_data)
        }
        
        # Try to determine if mulaw or pcm
        if len(audio_data) > 0:
            # Check first few bytes for common patterns
            if audio_data[:4] == b'RIFF':
                info["format"] = "wav"
                
                # Try to get more WAV info
                try:
                    with io.BytesIO(audio_data) as f:
                        with wave.open(f, 'rb') as wav:
                            info["channels"] = wav.getnchannels()
                            info["sample_width"] = wav.getsampwidth()
                            info["frame_rate"] = wav.getframerate()
                            info["n_frames"] = wav.getnframes()
                            info["duration"] = wav.getnframes() / wav.getframerate()
                except Exception as e:
                    logger.warning(f"Error extracting WAV info: {e}")
            else:
                # Rough guess based on values
                sample_values = np.frombuffer(audio_data[:100], dtype=np.uint8)
                if np.any(sample_values > 127):
                    info["format"] = "mulaw"
                else:
                    info["format"] = "pcm"
        
        return info
    
    @staticmethod
    def float32_to_pcm16(audio_data: np.ndarray) -> bytes:
        """
        Convert float32 audio to 16-bit PCM bytes.
        
        Args:
            audio_data: Audio data as numpy array (float32)
            
        Returns:
            Audio data as 16-bit PCM bytes
        """
        # Ensure audio is in [-1, 1] range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio_int16.tobytes()
    
    @staticmethod
    def pcm16_to_float32(audio_data: bytes) -> np.ndarray:
        """
        Convert 16-bit PCM bytes to float32 audio.
        
        Args:
            audio_data: Audio data as 16-bit PCM bytes
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Convert to numpy array
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32
            return audio_int16.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"Error converting PCM16 to float32: {e}")
            return np.array([], dtype=np.float32)
    
    @staticmethod
    def optimize_for_telephony(audio_data: bytes) -> bytes:
        """
        Optimize audio specifically for telephony transmission.
        This ensures the best possible audio quality over phone lines.
        
        Args:
            audio_data: Audio data as bytes (WAV or raw PCM)
            
        Returns:
            Optimized audio data
        """
        try:
            # Determine if WAV or raw PCM
            is_wav = audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE'
            
            if is_wav:
                # Extract PCM data and parameters from WAV
                with io.BytesIO(audio_data) as wav_io:
                    with wave.open(wav_io, 'rb') as wav_in:
                        sample_rate = wav_in.getframerate()
                        channels = wav_in.getnchannels()
                        width = wav_in.getsampwidth()
                        pcm_data = wav_in.readframes(wav_in.getnframes())
            else:
                # Assume 16-bit PCM at SAMPLE_RATE_AI
                pcm_data = audio_data
                sample_rate = SAMPLE_RATE_AI
                channels = 1
                width = 2
                
            # Multi-stage processing for telephony optimization
            
            # 1. Convert to mono if stereo
            if channels > 1:
                pcm_data = audioop.tomono(pcm_data, width, 0.5, 0.5)
            
            # 2. Increase volume slightly for better clarity over phone
            pcm_data = audioop.mul(pcm_data, width, 1.3)
            
            # 3. Apply dynamic range compression
            pcm_data = audioop.compress(pcm_data, width, 6, 2)
            
            # 4. Resample to 8 kHz (telephone quality)
            if sample_rate != SAMPLE_RATE_TWILIO:
                pcm_data, _ = audioop.ratecv(pcm_data, width, 1, sample_rate, SAMPLE_RATE_TWILIO, None)
            
            # 5. Convert to μ-law for Twilio
            mulaw_data = audioop.lin2ulaw(pcm_data, width)
            
            # Return μ-law encoded data ready for telephony
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error optimizing audio for telephony: {e}", exc_info=True)
            # Return original data if optimization fails
            return audio_data