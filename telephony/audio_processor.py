"""
Enhanced audio processing utilities for telephony integration with Google Cloud TTS.

Handles audio format conversion between Twilio and Voice AI Agent with improved
speech/noise discrimination.
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
    Optimized for Google Cloud TTS integration and speech/noise discrimination.
    
    Twilio uses 8kHz mulaw encoding, while our Voice AI uses 16kHz PCM.
    """
    
    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM for Voice AI with enhanced speech/noise discrimination.
        
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
            
            # Apply spectral subtraction for background noise reduction
            if len(audio_array) >= 512:  # Need enough samples for FFT
                # Use improved spectral subtraction
                audio_array = AudioProcessor._apply_spectral_subtraction(audio_array)
            
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
    def _apply_spectral_subtraction(audio_data: np.ndarray, frame_size: int = 512, overlap: int = 384) -> np.ndarray:
        """
        Apply spectral subtraction to reduce background noise.
        
        Args:
            audio_data: Audio data as numpy array
            frame_size: Frame size for STFT
            overlap: Overlap between frames
            
        Returns:
            Enhanced audio data
        """
        try:
            # Compute Short-Time Fourier Transform (STFT)
            f, t, Zxx = signal.stft(audio_data, fs=SAMPLE_RATE_AI, nperseg=frame_size, noverlap=overlap)
            
            # Get magnitude and phase components
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Estimate noise spectrum from lowest energy frames
            mag_energy = np.sum(mag**2, axis=0)
            sorted_indices = np.argsort(mag_energy)
            
            # Use first 10% of frames or at least 1 frame as noise estimate
            noise_frames = max(1, int(0.1 * len(sorted_indices)))
            noise_indices = sorted_indices[:noise_frames]
            
            # Compute noise estimate for each frequency bin
            noise_estimate = np.mean(mag[:, noise_indices], axis=1, keepdims=True)
            
            # Apply spectral subtraction with flooring to avoid negative values
            # and oversubtraction to reduce musical noise
            oversubtraction_factor = 1.5  # Subtract more noise than estimated
            spectral_floor = 0.01  # Minimum magnitude after subtraction
            
            # Apply subtraction with flooring
            mag_enhanced = np.maximum(
                mag - oversubtraction_factor * noise_estimate,
                spectral_floor * mag
            )
            
            # Reconstruct complex spectrum with original phase
            Zxx_enhanced = mag_enhanced * np.exp(1j * phase)
            
            # Inverse STFT to get time domain signal
            _, audio_enhanced = signal.istft(Zxx_enhanced, fs=SAMPLE_RATE_AI, nperseg=frame_size, noverlap=overlap)
            
            # Ensure same length as original signal
            if len(audio_enhanced) < len(audio_data):
                # Zero-pad if shorter
                audio_enhanced = np.pad(audio_enhanced, (0, len(audio_data) - len(audio_enhanced)))
            elif len(audio_enhanced) > len(audio_data):
                # Truncate if longer
                audio_enhanced = audio_enhanced[:len(audio_data)]
            
            return audio_enhanced
            
        except Exception as e:
            logger.error(f"Error applying spectral subtraction: {e}")
            return audio_data  # Return original signal if error
    
    @staticmethod
    def pcm_to_mulaw(pcm_data: bytes, source_sample_rate: int = None) -> bytes:
        """
        Convert PCM audio from Google Cloud TTS to mulaw for Twilio with enhanced quality.
        
        Args:
            pcm_data: PCM audio data
            source_sample_rate: Source sample rate (default: SAMPLE_RATE_AI)
            
        Returns:
            Mulaw audio data
        """
        try:
            # Check for extremely small data
            if len(pcm_data) < 320:  # Less than 20ms at 8kHz
                # Pad tiny chunks to avoid warnings
                padding_size = 320 - len(pcm_data)
                pcm_data = pcm_data + (b'\x00' * padding_size)
                logger.debug(f"Padded small audio chunk from {len(pcm_data)-padding_size} to {len(pcm_data)} bytes")
            
            # Try to detect input format and parameters
            detected_sample_rate = source_sample_rate or SAMPLE_RATE_AI
            
            # Convert to PCM if WAV
            if pcm_data[:4] == b'RIFF' and pcm_data[8:12] == b'WAVE':
                with io.BytesIO(pcm_data) as wav_io:
                    with wave.open(wav_io, 'rb') as wav_file:
                        detected_sample_rate = wav_file.getframerate()
                        n_channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        pcm_data = wav_file.readframes(wav_file.getnframes())
            
            # Ensure we have an even number of bytes for 16-bit samples
            if len(pcm_data) % 2 != 0:
                pcm_data = pcm_data + b'\x00'
            
            # Apply enhanced audio processing for telephony
            # First convert to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Apply compression for consistent volumes
            audio_array = AudioProcessor._apply_dynamic_compression(audio_array)
            
            # Convert back to bytes
            pcm_data = (audio_array * 32767.0).astype(np.int16).tobytes()
            
            # Resample to 8kHz for Twilio
            if detected_sample_rate != SAMPLE_RATE_TWILIO:
                try:
                    pcm_data_8k, _ = audioop.ratecv(
                        pcm_data, 2, 1, 
                        detected_sample_rate, 
                        SAMPLE_RATE_TWILIO, 
                        None
                    )
                except Exception as resample_error:
                    logger.error(f"Error resampling audio: {resample_error}")
                    pcm_data_8k = pcm_data
            else:
                pcm_data_8k = pcm_data
            
            # Convert to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_data_8k, 2)
            
            # Ensure minimum size for mulaw data
            if len(mulaw_data) < 160:  # Minimum 20ms chunk
                padding_needed = 160 - len(mulaw_data)
                mulaw_data = mulaw_data + bytes([0x7f] * padding_needed)  # Use 0x7f (silence) for padding
            
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error converting PCM to mulaw: {e}", exc_info=True)
            # Return silence rather than empty data
            return bytes([0x7f] * 160)  # Return 20ms of silence
    
    @staticmethod
    def _apply_dynamic_compression(audio_data: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to make telephone speech more consistent.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Compressed audio data
        """
        try:
            # Parameters for compression
            threshold = 0.3  # Compression threshold (-12dB)
            ratio = 0.5      # 2:1 compression ratio
            attack = 0.005   # 5ms attack time
            release = 0.05   # 50ms release time
            makeup_gain = 1.5  # Gain to apply after compression
            
            # Convert attack/release to samples
            attack_samples = int(attack * SAMPLE_RATE_AI)
            release_samples = int(release * SAMPLE_RATE_AI)
            
            # Initialize gain tracker
            envelope = np.zeros_like(audio_data)
            gain = np.ones_like(audio_data)
            
            # Simple envelope follower
            for i in range(1, len(audio_data)):
                # Current sample absolute value
                current_level = abs(audio_data[i])
                
                # Attack/release envelope detection
                if current_level > envelope[i-1]:
                    # Attack phase - fast rise
                    envelope[i] = envelope[i-1] + (current_level - envelope[i-1]) / attack_samples
                else:
                    # Release phase - slow fall
                    envelope[i] = envelope[i-1] + (current_level - envelope[i-1]) / release_samples
            
            # Apply compression curve
            for i in range(len(audio_data)):
                if envelope[i] > threshold:
                    # Above threshold, apply compression
                    above_threshold = envelope[i] - threshold
                    compressed = threshold + above_threshold * ratio
                    gain[i] = compressed / envelope[i]
                else:
                    # Below threshold, no compression
                    gain[i] = 1.0
            
            # Apply gain smoothing (additional pass)
            smoothed_gain = np.zeros_like(gain)
            smoothed_gain[0] = gain[0]
            
            # Simple lowpass filter on gain to reduce artifacts
            alpha = 0.9  # Smoothing factor
            for i in range(1, len(gain)):
                smoothed_gain[i] = alpha * smoothed_gain[i-1] + (1-alpha) * gain[i]
            
            # Apply compressed gain with makeup gain
            compressed_audio = audio_data * smoothed_gain * makeup_gain
            
            # Apply hard limiting to prevent clipping
            compressed_audio = np.clip(compressed_audio, -0.95, 0.95)
            
            return compressed_audio
            
        except Exception as e:
            logger.error(f"Error applying compression: {e}")
            return audio_data  # Return original if compression fails
    
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
        Using multi-stage processing for optimal speech/noise discrimination.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Enhanced audio data
        """
        try:
            # 1. Apply high-pass filter to remove low-frequency noise (below 100Hz)
            # Telephone lines often have low frequency hum
            b, a = signal.butter(4, 100/(SAMPLE_RATE_AI/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply band-pass filter for telephony freq range (300-3400 Hz)
            b, a = signal.butter(4, [300/(SAMPLE_RATE_AI/2), 3400/(SAMPLE_RATE_AI/2)], 'band')
            band_limited = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Apply spectral subtraction for noise reduction
            if len(band_limited) >= 512:
                clean_audio = AudioProcessor._apply_spectral_subtraction(band_limited)
            else:
                clean_audio = band_limited
            
            # 4. Apply mild compression for consistent volume
            compressed = AudioProcessor._apply_dynamic_compression(clean_audio)
            
            # 5. Apply pre-emphasis filter to boost higher frequencies (for speech clarity)
            pre_emphasis = np.append(compressed[0], compressed[1:] - 0.95 * compressed[:-1])
            
            # 6. Normalize audio level
            if np.max(np.abs(pre_emphasis)) > 0:
                normalized = pre_emphasis / np.max(np.abs(pre_emphasis)) * 0.95
            else:
                normalized = pre_emphasis
            
            return normalized
            
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
            
            # 2. Convert to numpy for enhanced processing
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 3. Apply dynamic compression for telephony
            audio_array = AudioProcessor._apply_dynamic_compression(audio_array)
            
            # 4. Convert back to bytes
            pcm_data = (audio_array * 32767.0).astype(np.int16).tobytes()
            
            # 5. Resample to 8 kHz (telephone quality)
            if sample_rate != SAMPLE_RATE_TWILIO:
                pcm_data, _ = audioop.ratecv(pcm_data, width, 1, sample_rate, SAMPLE_RATE_TWILIO, None)
            
            # 6. Convert to μ-law for Twilio
            mulaw_data = audioop.lin2ulaw(pcm_data, width)
            
            # Return μ-law encoded data ready for telephony
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error optimizing audio for telephony: {e}", exc_info=True)
            # Return original data if optimization fails
            return audio_data