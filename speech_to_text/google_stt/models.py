"""
Data models for Google Cloud Speech-to-Text integration.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class TranscriptionConfig(BaseModel):
    """Configuration for transcription requests."""
    
    language_code: Optional[str] = Field(
        default=None,
        description="Language code for transcription"
    )
    
    sample_rate_hertz: Optional[int] = Field(
        default=None,
        description="Audio sample rate in Hz"
    )
    
    encoding: Optional[str] = Field(
        default=None,
        description="Audio encoding format"
    )
    
    model: Optional[str] = Field(
        default=None,
        description="Recognition model to use"
    )
    
    use_enhanced: Optional[bool] = Field(
        default=None,
        description="Whether to use enhanced model"
    )
    
    enable_automatic_punctuation: Optional[bool] = Field(
        default=None,
        description="Whether to add punctuation"
    )
    
    enable_word_time_offsets: Optional[bool] = Field(
        default=None,
        description="Whether to include word timestamps"
    )
    
    enable_word_confidence: Optional[bool] = Field(
        default=None,
        description="Whether to include word confidence"
    )
    
    speech_contexts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Speech context for improved recognition"
    )
    
    profanity_filter: Optional[bool] = Field(
        default=None,
        description="Whether to filter profanity"
    )
    
    enable_separate_recognition_per_channel: Optional[bool] = Field(
        default=None,
        description="Whether to recognize each channel separately"
    )
    
    max_alternatives: Optional[int] = Field(
        default=None,
        description="Maximum number of alternatives to return"
    )

class TranscriptionResult(BaseModel):
    """Result of a transcription request."""
    
    transcription: str = Field(
        description="Transcribed text"
    )
    
    confidence: float = Field(
        default=0.0,
        description="Confidence score (0.0-1.0)"
    )
    
    words: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Word-level details"
    )
    
    alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative transcripts"
    )
    
    is_final: bool = Field(
        default=True,
        description="Whether this is a final result"
    )
    
    stability: float = Field(
        default=1.0,
        description="Stability of streaming result (0.0-1.0)"
    )
    
    duration: float = Field(
        default=0.0,
        description="Duration of audio in seconds"
    )