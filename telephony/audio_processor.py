"""
Enhanced audio processing utilities for telephony integration with Deepgram STT.

Handles audio format conversion between Twilio and Voice AI Agent.
"""
import audioop
import numpy as np
import logging
from typing import Tuple, Dict, Any
from scipy import signal

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio conversion between Twilio and Voice AI formats with improved noise handling.
    Optimized for Deepgram STT integration.
    
    Twilio uses 8kHz mulaw encoding, while our Voice AI uses 16kHz PCM.
    """
    
    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM for Voice AI with enhanced noise filtering.
        Optimized for Deepgram STT processing.
        
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
    def pcm_to_mulaw(pcm_data: bytes) -> bytes:
        """
        Convert PCM audio from Voice AI to mulaw for Twilio.
        
        Args:
            pcm_data: Audio data in PCM format
            
        Returns:
            Audio data in mulaw format
        """
        try:
            # Check if the data length is a multiple of 2 (for 16-bit samples)
            if len(pcm_data) % 2 != 0:
                # Pad with a zero byte to make it even
                pcm_data = pcm_data + b'\x00'
                logger.debug("Padded audio data to make even length")
            
            # Resample from 16kHz to 8kHz
            pcm_data_8k, _ = audioop.ratecv(
                pcm_data, 2, 1, 
                SAMPLE_RATE_AI, 
                SAMPLE_RATE_TWILIO, 
                None
            )
            
            # Convert to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_data_8k, 2)
            
            logger.debug(f"Converted {len(pcm_data)} bytes of PCM to {len(mulaw_data)} bytes of mulaw")
            
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error converting PCM to mulaw: {e}")
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
        Optimized for Deepgram STT processing.
        
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