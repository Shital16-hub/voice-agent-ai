"""
Enhanced audio processing utilities for telephony integration with improved
speech/noise discrimination and barge-in detection. This version simplifies
the audio processing to reduce dependency on AudioPreprocessor.
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
    Provides a simplified wrapper around AudioPreprocessor for barge-in detection
    while using direct processing for most operations.
    """
    
    def __init__(self):
        """Initialize the audio processor with minimal preprocessing."""
        # Create a minimal audio preprocessor just for barge-in detection
        # but don't use it for regular audio processing
        self.preprocessor = AudioPreprocessor(
            sample_rate=SAMPLE_RATE_AI,
            enable_barge_in=True,
            barge_in_threshold=0.03,  # Lower than default for better sensitivity
            min_speech_frames_for_barge_in=6,  # Fewer than default for faster response
            barge_in_cooldown_ms=1000,  # Shorter cooldown for more responsive barge-in
            enable_debug=False  # Disable debug logging
        )
        
        # Add direct barge-in detection without relying on complex preprocessing
        self.direct_barge_in_threshold = 0.025  # Even lower threshold for direct detection
        self.direct_barge_in_window_ms = 100  # Window size in ms for energy calculation
        self.last_barge_in_time = 0.0
        self.min_barge_in_interval = 0.5  # Minimum time between barge-in detections
        
        logger.info("Initialized AudioProcessor with enhanced speech preprocessing")
    
    def mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM for Voice AI with minimal preprocessing.
        
        Args:
            mulaw_data: Audio data in mulaw format
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Check if we have enough data to convert
            if len(mulaw_data) < 160:  # At least 20ms at 8kHz
                logger.debug(f"Small mulaw data: {len(mulaw_data)} bytes")
                # Pad to minimum size
                mulaw_data = mulaw_data + bytes([0x7f] * (160 - len(mulaw_data)))
            
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
            
            # Apply minimal preprocessing - just basic noise gate and normalization
            # Simple noise gate - very mild
            noise_threshold = 0.01  # Very low threshold
            audio_array = np.where(np.abs(audio_array) < noise_threshold, 0, audio_array)
            
            # Simple normalization if needed
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.95
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            # Return an empty array rather than raising an exception
            return np.array([], dtype=np.float32)
    
    def pcm_to_mulaw(self, pcm_data: bytes, source_sample_rate: int = None) -> bytes:
        """
        Convert PCM audio to mulaw for Twilio with minimal processing.
        
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
            
            # Apply minimal processing for telephony
            # Simple compression for consistent volumes
            if isinstance(pcm_data, bytes):
                # Convert to numpy array for processing
                audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Apply simple compression
                audio_array = self._apply_simple_compression(audio_array)
                
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
            logger.error(f"Error converting PCM to mulaw: {e}")
            # Return silence rather than empty data
            return bytes([0x7f] * 160)  # Return 20ms of silence
    
    def _apply_simple_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply simple compression to make telephone speech more consistent.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Compressed audio data
        """
        try:
            # Only process if we have data
            if len(audio_data) == 0:
                return audio_data
                
            # Simple compression parameters
            threshold = 0.3      # Threshold level
            ratio = 0.5          # 2:1 compression ratio
            makeup_gain = 1.5    # Makeup gain
            
            # Apply compression
            compressed = np.zeros_like(audio_data)
            for i in range(len(audio_data)):
                # Get absolute value
                level = abs(audio_data[i])
                
                # Apply compression if above threshold
                if level > threshold:
                    # Calculate gain reduction
                    above_threshold = level - threshold
                    compressed_amount = above_threshold * ratio
                    new_level = threshold + compressed_amount
                    
                    # Apply with original sign
                    compressed[i] = (new_level if audio_data[i] > 0 else -new_level) * makeup_gain
                else:
                    # Below threshold, apply makeup gain only
                    compressed[i] = audio_data[i] * makeup_gain
            
            # Apply hard limiting to prevent clipping
            compressed = np.clip(compressed, -0.98, 0.98)
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error applying compression: {e}")
            return audio_data  # Return original if compression fails
    
    def direct_check_for_barge_in(self, audio_data: np.ndarray) -> bool:
        """
        Direct barge-in detection without relying on complex AudioPreprocessor.
        
        This method performs simple energy-based detection to identify
        potential user speech during agent speech.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if barge-in is detected
        """
        # Check if agent is speaking
        if not self.preprocessor.agent_speaking:
            return False  # Only check for barge-in when agent is speaking
        
        try:
            # Calculate energy of the audio
            energy = np.mean(np.abs(audio_data))
            
            # Apply direct threshold
            if energy > self.direct_barge_in_threshold:
                # Check time since last barge-in to avoid multiple triggers
                current_time = time.time()
                if (current_time - self.last_barge_in_time) > self.min_barge_in_interval:
                    # Check time since agent started speaking - add a short delay
                    # to avoid detecting the agent's own speech
                    time_since_agent_started = (current_time - self.preprocessor.agent_speaking_start_time)
                    if time_since_agent_started > 0.8:  # 800ms delay
                        # Update last barge-in time
                        self.last_barge_in_time = current_time
                        
                        logger.info(f"Barge-in detected with direct method! Energy: {energy:.4f}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in direct barge-in detection: {e}")
            return False
    
    def contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Simplified speech detection that's more lenient.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if audio contains speech
        """
        try:
            # Calculate simple audio energy
            energy = np.mean(np.abs(audio_data))
            
            # Use a simple threshold that's lenient
            speech_energy_threshold = 0.02  # Very low threshold to catch most speech
            
            # Check if enough energy is present
            has_speech = energy > speech_energy_threshold
            
            # Add some basic frequency analysis to reduce false positives
            if has_speech and len(audio_data) > 512:
                try:
                    # Calculate spectrum
                    fft_data = np.abs(np.fft.rfft(audio_data))
                    freqs = np.fft.rfftfreq(len(audio_data), 1/SAMPLE_RATE_AI)
                    
                    # Calculate energy in speech frequencies (300-3400 Hz for telephony)
                    speech_freq_mask = (freqs >= 300) & (freqs <= 3400)
                    speech_energy = np.sum(fft_data[speech_freq_mask])
                    total_energy = np.sum(fft_data)
                    
                    if total_energy > 0:
                        speech_ratio = speech_energy / total_energy
                        # At least 30% of energy should be in speech frequencies
                        has_speech = has_speech and (speech_ratio > 0.3)
                except Exception as e:
                    logger.debug(f"Error in frequency analysis: {e}")
                    # Fall back to just energy detection
                    pass
            
            return has_speech
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            # Be conservative - assume speech is present
            return True
    
    def check_for_barge_in(self, audio_data: np.ndarray) -> bool:
        """
        Check for barge-in using both direct detection and AudioPreprocessor.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if barge-in is detected
        """
        # First try direct detection as it's faster
        if self.direct_check_for_barge_in(audio_data):
            return True
            
        # Fall back to the preprocessor's detection
        return self.preprocessor.check_for_barge_in(audio_data)
    
    def set_agent_speaking(self, is_speaking: bool) -> None:
        """
        Update the agent speaking state for barge-in detection.
        
        Args:
            is_speaking: Whether the agent is currently speaking
        """
        # Update the preprocessor's agent speaking state
        self.preprocessor.set_agent_speaking(is_speaking)
        
        # Also update local tracking for direct detection
        if is_speaking:
            # Agent started speaking - update the timestamp
            self.preprocessor.agent_speaking_start_time = time.time()
    
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
        self.last_barge_in_time = 0.0
        logger.info("Audio processor state reset")