"""
Audio processing utilities for telephony integration.

Handles audio format conversion between Twilio and Voice AI Agent.
"""
import audioop
import numpy as np
import logging
from typing import Tuple, Dict, Any

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio conversion between Twilio and Voice AI formats.
    
    Twilio uses 8kHz mulaw encoding, while our Voice AI uses 16kHz PCM.
    """
    
    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM for Voice AI.
        
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
            
            # Check audio levels
            audio_level = np.mean(np.abs(audio_array)) * 100
            logger.debug(f"Converted {len(mulaw_data)} bytes to {len(audio_array)} samples. Audio level: {audio_level:.1f}%")
            
            # Apply a gain if audio is very quiet (optional)
            if audio_level < 1.0:  # Very quiet audio
                audio_array = audio_array * min(5.0, 5.0/audio_level)
                logger.debug(f"Applied gain to quiet audio. New level: {np.mean(np.abs(audio_array)) * 100:.1f}%")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            raise
    
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
            raise
    
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
    def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if audio contains silence.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Silence threshold
            
        Returns:
            True if audio is considered silence
        """
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
        # Convert to numpy array
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32
        return audio_int16.astype(np.float32) / 32768.0