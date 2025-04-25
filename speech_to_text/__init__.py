"""
Speech-to-text module for the Voice AI Agent.

This module provides real-time streaming speech recognition using Whisper.cpp.
"""

import logging
from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult
from speech_to_text.streaming.chunker import AudioChunker, ChunkMetadata

__version__ = "0.1.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    "StreamingWhisperASR",
    "StreamingTranscriptionResult",
    "AudioChunker",
    "ChunkMetadata",
    "ConfigLoader",
]