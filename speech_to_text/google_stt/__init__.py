"""
Google Cloud Speech-to-Text client for Voice AI Agent.
"""

from .client import GoogleCloudSTT
from .streaming import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from .models import TranscriptionResult, TranscriptionConfig
from .exceptions import STTError, STTAPIError, STTStreamingError, STTConfigError, STTAudioError

__all__ = [
    'GoogleCloudSTT',
    'GoogleCloudStreamingSTT',
    'StreamingTranscriptionResult',
    'TranscriptionResult',
    'TranscriptionConfig',
    'STTError',
    'STTAPIError',
    'STTStreamingError',
    'STTConfigError',
    'STTAudioError'
]