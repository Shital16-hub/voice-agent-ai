"""
Google Cloud Speech-to-Text integration.
"""
from .client import GoogleCloudSTT
from .streaming import STTStreamer, SpeechDetector
from .config import config
from .exceptions import STTError, STTAPIError, STTStreamingError

__all__ = [
    'GoogleCloudSTT',
    'STTStreamer',
    'SpeechDetector',
    'config',
    'STTError',
    'STTAPIError',
    'STTStreamingError'
]