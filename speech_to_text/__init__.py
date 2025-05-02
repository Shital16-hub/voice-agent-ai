"""
Speech-to-text module for the Voice AI Agent.

This module provides real-time streaming speech recognition using Google Cloud's API.
"""

import logging
from speech_to_text.google_stt.client import GoogleCloudSTT
from speech_to_text.google_stt.streaming import STTStreamer, SpeechDetector
from speech_to_text.streamer import GoogleCloudStreamer, StreamingTranscriptionResult

__version__ = "0.2.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    "GoogleCloudSTT",
    "STTStreamer",
    "SpeechDetector",
    "GoogleCloudStreamer",
    "StreamingTranscriptionResult"
]