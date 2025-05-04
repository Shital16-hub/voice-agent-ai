"""
Configuration settings for the speech-to-text module.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class STTConfig(BaseSettings):
    """Configuration for Speech-to-Text module."""
    
    # Deepgram API settings
    deepgram_api_key: str = Field(
        default=os.getenv("DEEPGRAM_API_KEY", ""),
        description="Deepgram API Key for STT services"
    )
    
    # STT settings
    model_name: str = Field(
        default="general",
        description="Deepgram STT model to use"
    )
    
    language: str = Field(
        default="en-US",
        description="Language code for speech recognition"
    )
    
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    
    # Streaming settings
    interim_results: bool = Field(
        default=True,
        description="Whether to return interim results"
    )
    
    endpointing: str = Field(
        default="500",
        description="Endpointing in ms, or 'default'"
    )
    
    vad_events: bool = Field(
        default=True,
        description="Whether to return VAD events"
    )
    
    # Telephony optimizations
    utterance_end_ms: int = Field(
        default=500,
        description="Milliseconds of silence to consider an utterance complete"
    )
    
    keywords: list = Field(
        default=["price", "plan", "cost", "subscription", "service", "features", "support"],
        description="Keywords to boost in telephony context"
    )
    
    alternatives: int = Field(
        default=1,
        description="Number of alternative transcripts to return"
    )
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of STT results"
    )
    
    smart_format: bool = Field(
        default=True,
        description="Whether to apply smart formatting to numbers, dates, etc."
    )
    
    profanity_filter: bool = Field(
        default=False,
        description="Whether to filter profanity"
    )
    
    # Telephony-specific settings
    diarize: bool = Field(
        default=False,
        description="Whether to perform speaker diarization"
    )
    
    multichannel: bool = Field(
        default=False,
        description="Whether to treat audio as multichannel"
    )
    
    model_options: dict = Field(
        default={
            "tier": "enhanced",  # Use enhanced model for telephony
            "filler_words": False,  # Filter out um, uh, etc.
        },
        description="Additional model options"
    )
    
    class Config:
        env_prefix = "STT_"
        case_sensitive = False

# Create a global config instance
config = STTConfig()