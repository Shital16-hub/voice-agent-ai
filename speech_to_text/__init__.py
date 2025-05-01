"""
Speech-to-text module for the Voice AI Agent.

This module provides real-time streaming speech recognition using Google Cloud Speech-to-Text API.
"""

import logging
from speech_to_text.google_stt import (
    GoogleCloudSTT, 
    GoogleCloudStreamingSTT,
    StreamingTranscriptionResult,
    TranscriptionResult
)
from speech_to_text.stt_integration import STTIntegration

__version__ = "0.3.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    "GoogleCloudSTT",
    "GoogleCloudStreamingSTT", 
    "StreamingTranscriptionResult",
    "TranscriptionResult",
    "STTIntegration",
]