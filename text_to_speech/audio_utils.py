"""
Audio utilities for the text-to-speech module.

This module provides utilities for working with audio data,
including format conversion, streaming, and processing.
"""
import io
import os
import tempfile
import logging
from typing import Optional, Dict, Any, Union, Tuple, BinaryIO
import wave

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Utilities for processing audio data for delivery to telephony systems."""
    
    @staticmethod
    def get_audio_info(audio_data: bytes) -> Dict[str, Any]:
        """
        Extract information from audio data.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Dictionary with audio information (format, duration, etc.)
        """
        # Currently only supporting WAV format detection
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            with io.BytesIO(audio_data) as f:
                with wave.open(f, 'rb') as wav:
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    frame_rate = wav.getframerate()
                    n_frames = wav.getnframes()
                    duration = n_frames / frame_rate
                    
                    return {
                        'format': 'wav',
                        'channels': channels,
                        'sample_width': sample_width,
                        'frame_rate': frame_rate,
                        'n_frames': n_frames,
                        'duration': duration
                    }
        else:
            # For other formats, return basic info
            return {
                'format': 'unknown',
                'size': len(audio_data)
            }
    
    @staticmethod
    def convert_to_wav(
        audio_data: bytes,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None
    ) -> bytes:
        """
        Convert audio data to WAV format for telephony systems.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Target sample rate (Hz)
            channels: Target number of channels
            
        Returns:
            WAV audio data as bytes
        """
        # If we receive MP3 from Deepgram, we need to convert it
        # This requires ffmpeg to be installed
        try:
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
                mp3_file.write(audio_data)
                mp3_path = mp3_file.name
                
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_path = wav_file.name
            
            cmd = ['ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le']
            
            if sample_rate:
                cmd.extend(['-ar', str(sample_rate)])
            if channels:
                cmd.extend(['-ac', str(channels)])
                
            cmd.append(wav_path)
            
            # Run ffmpeg
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read the resulting WAV file
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
                
            # Clean up temporary files
            os.unlink(mp3_path)
            os.unlink(wav_path)
            
            return wav_data
            
        except ImportError:
            logger.error("ffmpeg not available for audio conversion")
            raise RuntimeError("ffmpeg is required for audio conversion")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise RuntimeError(f"Audio conversion error: {str(e)}")
    
    @staticmethod
    def split_audio(
        audio_data: bytes,
        chunk_size: int = 4096
    ) -> list:
        """
        Split audio data into manageable chunks for streaming.
        
        Args:
            audio_data: Audio data as bytes
            chunk_size: Size of each chunk in bytes
            
        Returns:
            List of audio data chunks
        """
        return [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
    
    @staticmethod
    def prepare_for_freeswitch(
        audio_data: bytes,
        sample_rate: int = 8000,
        channels: int = 1
    ) -> bytes:
        """
        Prepare audio data for FreeSWITCH.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Target sample rate for FreeSWITCH
            channels: Target number of channels for FreeSWITCH
            
        Returns:
            Audio data converted for FreeSWITCH
        """
        # Convert to WAV format suitable for FreeSWITCH
        return AudioProcessor.convert_to_wav(
            audio_data, 
            sample_rate=sample_rate, 
            channels=channels
        )