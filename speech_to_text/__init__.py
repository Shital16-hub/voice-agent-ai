"""
Speech-to-text module for the Voice AI Agent.

This module provides real-time streaming speech recognition using Deepgram's API.
"""

import logging
from speech_to_text.deepgram_stt import (
    DeepgramSTT, 
    DeepgramStreamingSTT,
    StreamingTranscriptionResult,
    TranscriptionResult
)
from speech_to_text.stt_integration import STTIntegration

__version__ = "0.2.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    "DeepgramSTT",
    "DeepgramStreamingSTT", 
    "StreamingTranscriptionResult",
    "TranscriptionResult",
    "STTIntegration",
]