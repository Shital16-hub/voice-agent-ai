"""
Enhanced audio processing utilities for telephony integration with improved
speech/noise discrimination and barge-in detection.

This module enhances the original AudioProcessor with more robust speech
detection algorithms specifically designed for telephony environments.
"""
import audioop
import numpy as np
import logging
import io
import wave
import time
from typing import Tuple, Dict, Any, Optional
from scipy import signal
from collections import deque

from telephony.audio_preprocessor import AudioPreprocessor
from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio conversion between Twilio and Voice AI formats with improved telephony performance.
    Integrates the new AudioPreprocessor for superior speech/noise discrimination.
    
    Twilio uses 8kHz mulaw encoding, while our Voice AI uses 16kHz PCM.
    """
    
    def __init__(self):
        """Initialize the audio processor with enhanced speech preprocessing."""
        # Create the dedicated audio preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=SAMPLE_RATE_AI,
            enable_barge_in=True,
            barge_in_threshold=0.045,  # Increased from the default
            min_speech_frames_for_barge_in=10,  # Increased from 5 to 10
            barge_in_cooldown_ms=2000,  # 2 second cooldown
            enable_debug=False  # Set to True for verbose logging
        )
        
        logger.info("Initialized AudioProcessor with enhanced speech preprocessing")
    
    def mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
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
                logger.debug(f"Small mulaw data: {len(mulaw_data)} bytes")
            
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
            
            # Apply enhanced audio preprocessing using the dedicated preprocessor
            audio_array = self.preprocessor.process_audio(audio_array)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            # Return an empty array rather than raising an exception
            return np.array([], dtype=np.float32)
    
    def pcm_to_mulaw(self, pcm_data: bytes, source_sample_rate: int = None) -> bytes:
        """
        Convert PCM audio to mulaw for Twilio with enhanced quality.
        
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
            audio_array = self._apply_dynamic_compression(audio_array)
            
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
    
    def _apply_dynamic_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to make telephone speech more consistent.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Compressed audio data
        """
        try:
            # Parameters for compression
            threshold = 0.25     # Lower threshold than original
            ratio = 0.4          # More aggressive compression (2.5:1)
            attack = 0.003       # 3ms attack time
            release = 0.05       # 50ms release time
            makeup_gain = 1.8    # Higher makeup gain for telephony
            
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
    
    def contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Determine if audio contains speech using the enhanced preprocessor.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if audio contains speech
        """
        # Use the dedicated preprocessor for speech detection
        return self.preprocessor.contains_speech(audio_data)
    
    def check_for_barge_in(self, audio_data: np.ndarray) -> bool:
        """
        Check if a user is interrupting (barging in) using the enhanced preprocessor.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if barge-in is detected
        """
        # Use the dedicated preprocessor for barge-in detection
        return self.preprocessor.check_for_barge_in(audio_data)
    
    def set_agent_speaking(self, is_speaking: bool) -> None:
        """
        Update the agent speaking state for barge-in detection.
        
        Args:
            is_speaking: Whether the agent is currently speaking
        """
        # Update the preprocessor's agent speaking state
        self.preprocessor.set_agent_speaking(is_speaking)
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
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
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
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
    
    def reset(self) -> None:
        """Reset all preprocessor state."""
        self.preprocessor.reset()
        logger.info("Audio processor state reset")