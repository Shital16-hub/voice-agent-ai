"""
Data models for the Deepgram Speech-to-Text integration.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class TranscriptionConfig(BaseModel):
    """Configuration for transcription requests."""
    
    model: Optional[str] = Field(
        default=None,
        description="Deepgram model to use"
    )
    
    language: Optional[str] = Field(
        default=None,
        description="Language code for transcription"
    )
    
    smart_format: Optional[bool] = Field(
        default=None,
        description="Whether to apply smart formatting to numbers, dates, etc."
    )
    
    filler_words: Optional[bool] = Field(
        default=None,
        description="Whether to include filler words like 'um', 'uh', etc."
    )
    
    profanity_filter: Optional[bool] = Field(
        default=None,
        description="Whether to filter profanity"
    )
    
    redact: Optional[List[str]] = Field(
        default=None,
        description="PII to redact (e.g., ['pci', 'ssn', 'numbers'])"
    )
    
    diarize: Optional[bool] = Field(
        default=None,
        description="Whether to diarize the audio (identify speakers)"
    )
    
    multichannel: Optional[bool] = Field(
        default=None,
        description="Whether to process each audio channel separately"
    )
    
    alternatives: Optional[int] = Field(
        default=None,
        description="Number of alternative transcripts to return"
    )
    
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords to boost recognition for"
    )
    
    search: Optional[List[str]] = Field(
        default=None,
        description="Terms to search for in the transcript"
    )
    
    replace: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Terms to replace in the transcript"
    )
    
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to send callback to upon completion"
    )
    
    utterance_end_ms: Optional[str] = Field(
        default=None,
        description="Silence duration in ms to consider an utterance complete"
    )
    
    endpointing: Optional[str] = Field(
        default=None,
        description="Endpointing in ms, or 'default'"
    )
    
    interim_results: Optional[bool] = Field(
        default=None,
        description="Whether to return interim results"
    )
    
    tier: Optional[str] = Field(
        default=None,
        description="API tier to use (base, enhanced, etc.)"
    )
    
    punctuate: Optional[bool] = Field(
        default=None,
        description="Whether to add punctuation"
    )

class TranscriptionResult(BaseModel):
    """Result of a transcription request."""
    
    text: str = Field(
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
    
    audio_duration: float = Field(
        default=0.0,
        description="Duration of the audio in seconds"
    )
    
    def get_words_with_times(self) -> List[Dict[str, Any]]:
        """Get words with their timestamps."""
        return [
            {
                "word": word.get("word", ""),
                "start": word.get("start", 0.0),
                "end": word.get("end", 0.0),
                "confidence": word.get("confidence", 0.0)
            }
            for word in self.words
        ]