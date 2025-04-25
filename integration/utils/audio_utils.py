"""
Audio utility functions for Voice AI Agent integration.

This module provides helper functions for audio processing,
format conversion, and streaming.
"""
import os
import io
import tempfile
import logging
import subprocess
from typing import Tuple, List, Optional, BinaryIO

import numpy as np

logger = logging.getLogger(__name__)

def convert_mp3_to_wav(
    mp3_data: bytes,
    target_sample_rate: int = 16000,
    target_channels: int = 1
) -> bytes:
    """
    Convert MP3 audio data to WAV format suitable for telephony systems.
    
    Args:
        mp3_data: MP3 audio data as bytes
        target_sample_rate: Target sample rate in Hz
        target_channels: Target number of audio channels
        
    Returns:
        WAV audio data as bytes
    """
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
            mp3_file.write(mp3_data)
            mp3_path = mp3_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_path = wav_file.name
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', mp3_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(target_sample_rate),
            '-ac', str(target_channels),
            '-y',  # Overwrite output file if it exists
            wav_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Read the output file
        with open(wav_path, 'rb') as f:
            wav_data = f.read()
        
        # Clean up temporary files
        os.unlink(mp3_path)
        os.unlink(wav_path)
        
        return wav_data
    
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise RuntimeError(f"Error converting MP3 to WAV: {e.stderr}")
    except Exception as e:
        logger.error(f"Error in convert_mp3_to_wav: {str(e)}")
        raise

def prepare_audio_for_telephony(
    audio_data: bytes,
    format: str = 'mp3',
    target_sample_rate: int = 8000,
    target_channels: int = 1
) -> bytes:
    """
    Prepare audio data for telephony systems.
    
    Converts audio to the format and parameters expected by
    telephony systems like FreeSWITCH.
    
    Args:
        audio_data: Audio data as bytes
        format: Source format ('mp3', 'wav', etc.)
        target_sample_rate: Target sample rate in Hz
        target_channels: Target number of audio channels
        
    Returns:
        Processed audio data as bytes
    """
    try:
        if format.lower() == 'mp3':
            return convert_mp3_to_wav(
                audio_data,
                target_sample_rate=target_sample_rate,
                target_channels=target_channels
            )
        elif format.lower() == 'wav':
            # For WAV, we might need to resample
            # This is a simplified example - in a real implementation,
            # you'd check if resampling is needed
            return audio_data
        else:
            logger.warning(f"Unsupported format: {format}. Returning audio as-is.")
            return audio_data
    
    except Exception as e:
        logger.error(f"Error preparing audio for telephony: {str(e)}")
        raise

def get_audio_info(audio_data: bytes) -> dict:
    """
    Extract information about audio data.
    
    Args:
        audio_data: Audio data as bytes
        
    Returns:
        Dictionary with audio information
    """
    info = {
        'size': len(audio_data),
        'format': 'unknown'
    }
    
    # Check if it's a WAV file
    if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
        info['format'] = 'wav'
        
        # Extract more WAV info
        try:
            with io.BytesIO(audio_data) as f:
                import wave
                with wave.open(f, 'rb') as wav:
                    info['channels'] = wav.getnchannels()
                    info['sample_width'] = wav.getsampwidth()
                    info['sample_rate'] = wav.getframerate()
                    info['frames'] = wav.getnframes()
                    info['duration'] = info['frames'] / info['sample_rate']
        except Exception as e:
            logger.warning(f"Error extracting WAV info: {str(e)}")
    
    # Check if it's an MP3 file
    elif audio_data[:3] == b'ID3' or (audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0):
        info['format'] = 'mp3'
        
        # MP3 duration calculation would require additional libraries
    
    return info

def split_audio_into_chunks(
    audio_data: bytes,
    chunk_duration_ms: int = 100,
    format: str = 'wav',
    sample_rate: int = 16000,
    channels: int = 1
) -> List[bytes]:
    """
    Split audio data into time-based chunks.
    
    Args:
        audio_data: Audio data as bytes
        chunk_duration_ms: Chunk duration in milliseconds
        format: Audio format ('wav', 'mp3', etc.)
        sample_rate: Sample rate in Hz
        channels: Number of channels
        
    Returns:
        List of audio data chunks
    """
    # This is a simplified implementation
    # A real implementation would need to handle audio format specifics
    
    # For WAV, we can do time-based chunking
    if format.lower() == 'wav':
        try:
            with io.BytesIO(audio_data) as f:
                import wave
                with wave.open(f, 'rb') as wav:
                    # Get WAV properties
                    actual_channels = wav.getnchannels()
                    actual_sample_width = wav.getsampwidth()
                    actual_sample_rate = wav.getframerate()
                    
                    # Calculate chunk size in frames
                    chunk_size_frames = int(actual_sample_rate * chunk_duration_ms / 1000)
                    
                    # Read all audio data
                    frames = wav.readframes(wav.getnframes())
                    
                    # Calculate frame size in bytes
                    frame_size = actual_channels * actual_sample_width
                    
                    # Calculate chunk size in bytes
                    chunk_size_bytes = chunk_size_frames * frame_size
                    
                    # Split into chunks
                    chunks = []
                    for i in range(0, len(frames), chunk_size_bytes):
                        chunk_frames = frames[i:i+chunk_size_bytes]
                        
                        # Create a new WAV file in memory
                        chunk_io = io.BytesIO()
                        with wave.open(chunk_io, 'wb') as chunk_wav:
                            chunk_wav.setnchannels(actual_channels)
                            chunk_wav.setsampwidth(actual_sample_width)
                            chunk_wav.setframerate(actual_sample_rate)
                            chunk_wav.writeframes(chunk_frames)
                        
                        chunks.append(chunk_io.getvalue())
                    
                    return chunks
        except Exception as e:
            logger.error(f"Error splitting WAV: {str(e)}")
            raise
    
    # For MP3, splitting is more complex and would require additional libraries
    # For simplicity, just return the full audio as a single chunk
    logger.warning(f"Splitting {format} format not implemented. Returning single chunk.")
    return [audio_data]

def create_silent_audio(
    duration_ms: int = 500,
    sample_rate: int = 8000,
    channels: int = 1
) -> bytes:
    """
    Create silent audio of specified duration.
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        channels: Number of channels
        
    Returns:
        WAV audio data as bytes
    """
    # Calculate number of frames
    num_frames = int(sample_rate * duration_ms / 1000)
    
    # Create silent audio data (all zeros)
    audio_data = np.zeros(num_frames * channels, dtype=np.int16)
    
    # Convert to WAV
    with io.BytesIO() as f:
        import wave
        with wave.open(f, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_data.tobytes())
        
        return f.getvalue()