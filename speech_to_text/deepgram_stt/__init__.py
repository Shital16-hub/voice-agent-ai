"""
Deepgram Speech-to-Text client for Voice AI Agent.
"""

from .client import DeepgramSTT
from .streaming import DeepgramStreamingSTT, StreamingTranscriptionResult
from .models import TranscriptionResult, TranscriptionConfig
from .exceptions import STTError, STTAPIError, STTStreamingError, STTConfigError, STTAudioError

__all__ = [
    'DeepgramSTT',
    'DeepgramStreamingSTT',
    'StreamingTranscriptionResult',
    'TranscriptionResult',
    'TranscriptionConfig',
    'STTError',
    'STTAPIError',
    'STTStreamingError',
    'STTConfigError',
    'STTAudioError'
]