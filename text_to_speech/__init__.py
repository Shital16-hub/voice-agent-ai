"""
Text-to-Speech module for Voice AI Agent.

This module provides functionality for converting text to speech
using the Google Cloud TTS API, optimized for real-time voice applications.
"""

from .google_tts import GoogleCloudTTS
from .streaming import TTSStreamer, RealTimeResponseHandler
from .audio_utils import AudioProcessor
from .config import config, TTSConfig
from .exceptions import TTSError, TTSAPIError, TTSStreamingError, TTSConfigError, TTSAudioError

__all__ = [
    'GoogleCloudTTS',
    'TTSStreamer',
    'RealTimeResponseHandler',
    'AudioProcessor',
    'config',
    'TTSConfig',
    'TTSError',
    'TTSAPIError',
    'TTSStreamingError', 
    'TTSConfigError',
    'TTSAudioError'
]