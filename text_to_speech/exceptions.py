"""
Exceptions for the text-to-speech module.
"""

class TTSError(Exception):
    """Base exception for TTS module errors."""
    pass

class TTSAPIError(TTSError):
    """Exception for API-related errors."""
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