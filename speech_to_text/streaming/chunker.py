"""
Audio chunking utilities for streaming speech recognition.
"""

import numpy as np
import collections
from typing import Dict, List, Tuple, Optional, Deque
import logging

logger = logging.getLogger(__name__)

class AudioChunker:
    """
    Manages audio chunking for streaming recognition.
    
    This class handles the buffering and chunking of incoming audio data
    to ensure optimal processing by the speech recognition model.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size_ms: int = 1000,
        overlap_ms: int = 200,
        silence_threshold: float = 0.01,
        min_silence_ms: int = 500,
        max_chunk_size_ms: int = 30000,
    ):
        """
        Initialize AudioChunker.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size_ms: Size of each chunk in milliseconds
            overlap_ms: Overlap between chunks in milliseconds
            silence_threshold: Amplitude threshold below which audio is considered silence
            min_silence_ms: Minimum silence duration to consider a chunk boundary
            max_chunk_size_ms: Maximum chunk size in milliseconds
        """
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)
        self.overlap = int(sample_rate * overlap_ms / 1000)
        self.silence_threshold = silence_threshold
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.max_chunk_size = int(sample_rate * max_chunk_size_ms / 1000)
        
        # Buffer for incoming audio
        self.buffer: Deque[float] = collections.deque(maxlen=self.max_chunk_size)
        self.processed_samples = 0
        
        logger.info(
            f"Initialized AudioChunker with chunk_size={self.chunk_size} samples, "
            f"overlap={self.overlap} samples, silence_threshold={self.silence_threshold}"
        )
    
    def add_audio(self, audio: np.ndarray) -> bool:
        """
        Add audio data to the buffer.
        
        Args:
            audio: Audio data as a numpy array
            
        Returns:
            True if enough data is available for processing
        """
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure audio is mono
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        
        # Add to buffer
        self.buffer.extend(audio.flatten())
        
        # Check if we have enough data for a chunk
        return len(self.buffer) >= self.chunk_size
    
    def get_chunk(self, force: bool = False) -> Optional[np.ndarray]:
        """
        Get the next chunk from the buffer if available.
        
        Args:
            force: If True, return whatever data is in the buffer even if smaller than chunk_size
            
        Returns:
            Audio chunk as numpy array, or None if not enough data
        """
        if len(self.buffer) < self.chunk_size and not force:
            return None
        
        # Convert buffer to numpy array for processing
        audio_data = np.array(list(self.buffer), dtype=np.float32)
        
        # Find optimal chunk boundary (at silence if possible)
        chunk_boundary = self._find_chunk_boundary(audio_data)
        
        # Extract chunk
        chunk = audio_data[:chunk_boundary]
        
        # Update buffer: remove chunk but keep overlap
        overlap_start = max(0, chunk_boundary - self.overlap)
        new_buffer = audio_data[overlap_start:].tolist()
        self.buffer.clear()
        self.buffer.extend(new_buffer)
        
        # Update processed samples count
        self.processed_samples += chunk_boundary - self.overlap
        
        return chunk
    
    def get_final_chunk(self) -> Optional[np.ndarray]:
        """
        Get any remaining audio from the buffer as a final chunk.
        
        Returns:
            np.ndarray: The remaining audio samples, or None if buffer is empty
        """
        if len(self.buffer) == 0:
            return None
            
        # Convert buffer to numpy array
        final_chunk = np.array(list(self.buffer), dtype=np.float32)
        
        # Clear the buffer
        self.buffer.clear()
        
        # Update processed samples count
        self.processed_samples += len(final_chunk)
        
        return final_chunk
    
    def _find_chunk_boundary(self, audio: np.ndarray) -> int:
        """
        Find the optimal chunk boundary, preferably at a silence.
        
        Args:
            audio: Audio data
            
        Returns:
            Index of the optimal chunk boundary
        """
        # Default to chunk size if buffer is not much larger
        if len(audio) < self.chunk_size + self.min_silence_samples:
            return min(self.chunk_size, len(audio))
        
        # Calculate audio energy
        energy = np.abs(audio)
        
        # Look for silence in the expected region
        search_start = self.chunk_size - self.min_silence_samples
        search_end = min(len(audio), self.chunk_size + self.min_silence_samples)
        
        # Find regions below silence threshold
        silence_mask = energy[search_start:search_end] < self.silence_threshold
        
        # Find continuous silence segments
        silence_indices = np.where(silence_mask)[0] + search_start
        if len(silence_indices) > 0:
            # Find continuous segments
            segments = np.split(silence_indices, np.where(np.diff(silence_indices) > 1)[0] + 1)
            
            # Keep only segments that are long enough
            valid_segments = [seg for seg in segments if len(seg) >= self.min_silence_samples]
            
            if valid_segments:
                # Use the middle of the first valid silence segment
                silence_pos = valid_segments[0][len(valid_segments[0]) // 2]
                logger.debug(f"Found silence at position {silence_pos}")
                return silence_pos
        
        # If no suitable silence found, use the default chunk size
        return self.chunk_size
    
    def reset(self):
        """Reset the chunker state."""
        self.buffer.clear()
        self.processed_samples = 0

class ChunkMetadata:
    """Metadata for audio chunks."""
    
    def __init__(
        self,
        chunk_id: int,
        start_sample: int,
        end_sample: int,
        sample_rate: int = 16000,
        is_first_chunk: bool = False,
        contains_speech: bool = True
    ):
        """
        Initialize chunk metadata.
        
        Args:
            chunk_id: Unique identifier for the chunk
            start_sample: Start sample index in the overall stream
            end_sample: End sample index in the overall stream
            sample_rate: Sample rate of the audio
            is_first_chunk: Whether this is the first chunk in a new stream
            contains_speech: Whether the chunk is likely to contain speech
        """
        self.chunk_id = chunk_id
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.sample_rate = sample_rate
        self.is_first_chunk = is_first_chunk
        self.contains_speech = contains_speech
        
    @property
    def start_time(self) -> float:
        """Get start time in seconds."""
        return self.start_sample / self.sample_rate
    
    @property
    def end_time(self) -> float:
        """Get end time in seconds."""
        return self.end_sample / self.sample_rate
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return (self.end_sample - self.start_sample) / self.sample_rate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "is_first_chunk": self.is_first_chunk,
            "contains_speech": self.contains_speech
        }