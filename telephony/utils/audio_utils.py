"""
Audio utilities for telephony integration.
"""
import audioop
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

def mulaw_to_pcm(mulaw_data: bytes, sample_width: int = 2) -> bytes:
    """
    Convert μ-law audio to PCM.
    
    Args:
        mulaw_data: μ-law audio data
        sample_width: Sample width in bytes (default: 2 for 16-bit PCM)
        
    Returns:
        PCM audio data
    """
    return audioop.ulaw2lin(mulaw_data, sample_width)

def pcm_to_mulaw(pcm_data: bytes, sample_width: int = 2) -> bytes:
    """
    Convert PCM audio to μ-law.
    
    Args:
        pcm_data: PCM audio data
        sample_width: Sample width in bytes (default: 2 for 16-bit PCM)
        
    Returns:
        μ-law audio data
    """
    return audioop.lin2ulaw(pcm_data, sample_width)

def resample_audio(pcm_data: bytes, orig_rate: int, target_rate: int, 
                  width: int = 2, channels: int = 1) -> bytes:
    """
    Resample audio to target sample rate.
    
    Args:
        pcm_data: PCM audio data
        orig_rate: Original sample rate
        target_rate: Target sample rate
        width: Sample width in bytes
        channels: Number of channels
        
    Returns:
        Resampled audio data
    """
    # Use audioop for efficient resampling
    resampled_data, _ = audioop.ratecv(pcm_data, width, channels, 
                                      orig_rate, target_rate, None)
    return resampled_data

def convert_twilio_audio(audio_data: bytes) -> np.ndarray:
    """
    Convert Twilio μ-law audio to numpy array.
    
    Args:
        audio_data: μ-law audio data from Twilio
        
    Returns:
        Audio data as numpy array (float32)
    """
    # Convert μ-law to 16-bit PCM
    pcm_data = mulaw_to_pcm(audio_data)
    
    # Convert to numpy array
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
    
    # Normalize to float32 in [-1.0, 1.0] range
    audio_array = audio_array.astype(np.float32) / 32768.0
    
    return audio_array

def prepare_audio_for_twilio(audio_data: Union[bytes, np.ndarray], 
                            source_rate: int = 16000) -> bytes:
    """
    Prepare audio data for Twilio.
    
    Args:
        audio_data: Audio data as bytes or numpy array
        source_rate: Source sample rate
        
    Returns:
        μ-law audio data ready for Twilio
    """
    # Convert numpy array to bytes if needed
    if isinstance(audio_data, np.ndarray):
        # Ensure float32 in [-1.0, 1.0] range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    else:
        audio_bytes = audio_data
    
    # Resample to 8kHz if needed
    if source_rate != 8000:
        audio_bytes = resample_audio(audio_bytes, source_rate, 8000)
    
    # Convert to μ-law
    mulaw_audio = pcm_to_mulaw(audio_bytes)
    
    return mulaw_audio