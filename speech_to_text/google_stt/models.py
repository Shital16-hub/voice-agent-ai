"""
Data models for the Google Cloud Speech-to-Text integration.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class TranscriptionConfig(BaseModel):
    """Configuration for transcription requests."""
    
    language_code: str = Field(
        default="en-US",
        description="Language code for transcription"
    )
    
    enable_automatic_punctuation: bool = Field(
        default=True,
        description="Whether to add punctuation"
    )
    
    model: str = Field(
        default="phone_call",
        description="Google Cloud Speech-to-Text model to use"
    )
    
    use_enhanced: bool = Field(
        default=True,
        description="Whether to use enhanced models for better quality"
    )
    
    sample_rate_hertz: int = Field(
        default=8000,
        description="Sample rate of the audio"
    )
    
    encoding: str = Field(
        default="LINEAR16",
        description="Audio encoding format"
    )
    
    enable_word_time_offsets: bool = Field(
        default=True,
        description="Whether to include timing information for words"
    )
    
    alternative_language_codes: Optional[List[str]] = Field(
        default=None,
        description="Alternative language codes for multi-language transcription"
    )
    
    speech_contexts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Speech context for improved recognition of specific phrases"
    )
    
    audio_channel_count: int = Field(
        default=1,
        description="Number of audio channels (1 for mono, 2 for stereo)"
    )
    
    enable_separate_recognition_per_channel: bool = Field(
        default=False,
        description="Whether to recognize each channel separately"
    )
    
    max_alternatives: int = Field(
        default=1,
        description="Maximum number of alternative transcriptions to return"
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
                "start": word.get("start_time", 0.0),
                "end": word.get("end_time", 0.0),
                "confidence": word.get("confidence", 0.0)
            }
            for word in self.words
        ]