"""
Audio utilities for the speech-to-text module.
"""
import numpy as np
import io
import wave
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def load_audio_file(
    file_path: str,
    target_sr: int = 8000,
    convert_to_mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to appropriate format.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        convert_to_mono: Whether to convert to mono
        normalize: Whether to normalize audio
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Read WAV file
        with wave.open(file_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            orig_sr = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            
            # Convert to numpy array
            if sample_width == 1:
                audio = np.frombuffer(frames, dtype=np.uint8)
                audio = audio.astype(np.float32) / 128.0 - 1.0
            elif sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(frames, dtype=np.int32)
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Reshape for multi-channel audio
            if channels > 1:
                audio = audio.reshape(-1, channels)
    except (wave.Error, IOError, ValueError) as e:
        logger.error(f"Error loading WAV file: {e}")
        raise
    
    # Convert to mono if needed
    if convert_to_mono and channels > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if orig_sr != target_sr:
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            audio = signal.resample(audio, num_samples)
        except ImportError:
            logger.warning("scipy not installed, using linear interpolation for resampling")
            # Simple linear interpolation
            indices = np.linspace(0, len(audio) - 1, int(len(audio) * target_sr / orig_sr))
            indices = indices.astype(np.int32)
            audio = audio[indices]
    
    # Normalize if needed
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    
    return audio, target_sr