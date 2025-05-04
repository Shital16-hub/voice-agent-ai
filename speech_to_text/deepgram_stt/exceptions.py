"""
Exceptions for the Deepgram Speech-to-Text integration.
"""

class STTError(Exception):
    """Base exception for STT module errors."""
    pass

class STTAPIError(STTError):
    """Exception for API-related errors."""
    pass

class STTStreamingError(STTError):
    """Exception for streaming-related errors."""
    pass

class STTConfigError(STTError):
    """Exception for configuration errors."""
    pass

class STTAudioError(STTError):
    """Exception for audio processing errors."""
    pass