"""
Audio utilities for the speech-to-text module.

Enhanced to support Google Cloud Speech-to-Text requirements.
"""

import numpy as np
import io
import wave
import logging
from typing import Tuple, Optional, Union
from scipy import signal

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
        import librosa
        return librosa.resample(
            y=audio, 
            orig_sr=orig_sr, 
            target_sr=target_sr
        )
    except ImportError:
        logger.warning(
            "librosa not installed, using scipy for resampling. "
            "For better quality, install librosa."
        )
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            resampled = signal.resample(audio, num_samples)
            return resampled
        except ImportError:
            raise ImportError(
                "Neither librosa nor scipy is installed. "
                "Please install one of them for audio resampling."
            )

def load_audio_file(
    file_path: str,
    target_sr: int = 16000,
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
        import soundfile as sf
        audio, orig_sr = sf.read(file_path)
    except ImportError:
        logger.warning(
            "soundfile not installed, using scipy.io.wavfile. "
            "For better format support, install soundfile."
        )
        try:
            from scipy.io import wavfile
            orig_sr, audio = wavfile.read(file_path)
        except ImportError:
            raise ImportError(
                "Neither soundfile nor scipy is installed. "
                "Please install one of them for audio loading."
            )
    
    # Convert to mono if needed
    if convert_to_mono and len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    
    # Normalize if needed
    if normalize:
        audio = normalize_audio(audio)
    
    # Resample if needed
    if orig_sr != target_sr:
        audio = resample_audio(audio, orig_sr, target_sr)
    
    return audio, target_sr

def preprocess_audio_for_google_cloud(audio: np.ndarray) -> np.ndarray:
    """
    Preprocess audio specifically for Google Cloud Speech API.
    
    Args:
        audio: Audio data as numpy array
        
    Returns:
        Preprocessed audio data
    """
    try:
        # 1. High-pass filter (remove frequencies below 80Hz)
        b, a = signal.butter(4, 80/(16000/2), 'highpass')
        audio = signal.filtfilt(b, a, audio)
        
        # 2. Bandpass filter for speech frequencies (300-3400 Hz)
        b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
        audio = signal.filtfilt(b, a, audio)
        
        # 3. Apply pre-emphasis filter to boost higher frequencies
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 4. Normalize audio level
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * 0.9 / max_val
            
        return audio
    except Exception as e:
        logger.error(f"Error in audio preprocessing: {e}")
        return audio  # Return original audio if processing fails

def convert_audio_format(
    audio: np.ndarray,
    output_format: str = 'LINEAR16'
) -> bytes:
    """
    Convert audio to a format compatible with Google Cloud Speech API.
    
    Args:
        audio: Audio data as numpy array
        output_format: Output format code ('LINEAR16', 'FLAC', etc.)
        
    Returns:
        Formatted audio data as bytes
    """
    if output_format == 'LINEAR16':
        # Convert to 16-bit PCM
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
        return audio_int16.tobytes()
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def detect_voice_activity(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Detect if audio segment contains voice activity.
    
    Args:
        audio: Audio data as numpy array
        threshold: Energy threshold for voice detection
        
    Returns:
        True if voice activity is detected
    """
    # Calculate energy
    energy = np.mean(np.abs(audio))
    
    # Calculate zero-crossing rate (helps distinguish speech from noise)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
    
    # Detect speech based on energy and ZCR
    is_speech = (energy > threshold) and (zero_crossings > 0.01) and (zero_crossings < 0.15)
    
    return is_speech

def segment_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    segment_length_ms: int = 500,
    overlap_ms: int = 100
) -> list:
    """
    Segment audio into overlapping chunks for streaming processing.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        segment_length_ms: Segment length in milliseconds
        overlap_ms: Overlap between segments in milliseconds
        
    Returns:
        List of audio segments
    """
    # Calculate segment and hop sizes in samples
    segment_size = int(sample_rate * segment_length_ms / 1000)
    hop_size = int(sample_rate * (segment_length_ms - overlap_ms) / 1000)
    
    # Create segments
    segments = []
    for i in range(0, len(audio) - segment_size + 1, hop_size):
        segment = audio[i:i+segment_size]
        segments.append(segment)
    
    # Add final segment if needed
    if len(audio) > i + hop_size:
        segments.append(audio[i+hop_size:])
    
    return segments