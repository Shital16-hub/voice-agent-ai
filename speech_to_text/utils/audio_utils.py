"""
Audio utilities for the speech-to-text module.
"""
import numpy as np
import io
import wave
import logging
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to float32 in [-1.0, 1.0] range.
    
    Args:
        audio: Input audio array
        
    Returns:
        Normalized audio array
    """
    if audio.dtype == np.float32 and np.max(np.abs(audio)) <= 1.0:
        return audio
    
    # Convert to float32 if needed
    if audio.dtype != np.float32:
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0
        else:
            audio = audio.astype(np.float32)
    
    # Normalize to [-1.0, 1.0] if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    return audio

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to mono.
    
    Args:
        audio: Input audio array
        
    Returns:
        Mono audio array
    """
    if len(audio.shape) == 1:
        return audio
    
    if len(audio.shape) == 2 and audio.shape[1] > 1:
        return np.mean(audio, axis=1)
    
    return audio

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled
    except ImportError:
        raise ImportError(
            "scipy is not installed. Please install scipy for audio resampling."
        )

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
        audio = resample_audio(audio, orig_sr, target_sr)
    
    # Normalize if needed
    if normalize:
        audio = normalize_audio(audio)
    
    return audio, target_sr

def audio_bytes_to_array(
    audio_bytes: bytes,
    sample_width: int = 2,
    channels: int = 1,
    sample_rate: int = 8000,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Convert raw audio bytes to numpy array.
    
    Args:
        audio_bytes: Raw audio bytes
        sample_width: Sample width in bytes (1=8bit, 2=16bit, 4=32bit)
        channels: Number of audio channels
        sample_rate: Audio sample rate
        normalize: Whether to normalize audio
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Create a wave file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    
    # Read the wave file
    wav_buffer.seek(0)
    with wave.open(wav_buffer, 'rb') as wav_file:
        # Get audio data as bytes
        audio_bytes = wav_file.readframes(wav_file.getnframes())
        
        # Convert to numpy array
        if sample_width == 1:
            audio = np.frombuffer(audio_bytes, dtype=np.uint8)
            audio = (audio.astype(np.float32) - 128) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(audio_bytes, dtype=np.int32)
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Reshape for multi-channel audio
        if channels > 1:
            audio = audio.reshape(-1, channels)
        
        # Normalize if needed
        if normalize and np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
    
    return audio, sample_rate

def mulaw_to_linear(mulaw_data: bytes) -> bytes:
    """
    Convert μ-law audio to 16-bit PCM.
    
    Args:
        mulaw_data: μ-law audio data
        
    Returns:
        16-bit PCM audio data
    """
    try:
        import audioop
        return audioop.ulaw2lin(mulaw_data, 2)
    except ImportError:
        raise ImportError("audioop module is required for μ-law conversion")

def linear_to_mulaw(linear_data: bytes) -> bytes:
    """
    Convert 16-bit PCM to μ-law audio.
    
    Args:
        linear_data: 16-bit PCM audio data
        
    Returns:
        μ-law audio data
    """
    try:
        import audioop
        return audioop.lin2ulaw(linear_data, 2)
    except ImportError:
        raise ImportError("audioop module is required for μ-law conversion")