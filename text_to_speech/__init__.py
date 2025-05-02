"""
Text-to-Speech module for Voice AI Agent.

This module provides functionality for converting text to speech
using the Deepgram TTS API, optimized for real-time voice applications.
"""

from .deepgram_tts import DeepgramTTS
from .streaming import TTSStreamer, RealTimeResponseHandler
from .audio_utils import AudioProcessor
from .config import config, TTSConfig
from .exceptions import TTSError, TTSAPIError, TTSStreamingError, TTSConfigError, TTSAudioError

__all__ = [
    'DeepgramTTS',
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