"""
Exceptions for the text-to-speech module with Google Cloud TTS.
"""

class TTSError(Exception):
    """Base exception for TTS module errors."""
    pass

class TTSAPIError(TTSError):
    """Exception for Google Cloud API-related errors."""
    pass

class TTSStreamingError(TTSError):
    """Exception for streaming-related errors."""
    pass

class TTSConfigError(TTSError):
    """Exception for configuration errors."""
    pass

class TTSAudioError(TTSError):
    """Exception for audio processing errors."""
    pass