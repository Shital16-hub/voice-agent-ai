"""
Speech-to-text module for the Voice AI Agent.
"""
import logging
from speech_to_text.google_stt.client import GoogleCloudSTT
from speech_to_text.google_stt.streaming import STTStreamer, SpeechDetector

__version__ = "0.2.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    "GoogleCloudSTT",
    "STTStreamer",
    "SpeechDetector"
]