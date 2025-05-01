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
    
    # Google STT API settings
    google_application_credentials: str = Field(
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        description="Path to Google Cloud credentials JSON file"
    )
    
    # STT settings
    model_name: str = Field(
        default="phone_call",
        description="Google STT model to use"
    )
    
    language: str = Field(
        default="en-US",
        description="Language code for speech recognition"
    )
    
    sample_rate: int = Field(
        default=8000,
        description="Audio sample rate in Hz"
    )
    
    # Streaming settings
    interim_results: bool = Field(
        default=True,
        description="Whether to return interim results"
    )
    
    use_enhanced: bool = Field(
        default=True, 
        description="Whether to use enhanced models"
    )
    
    # Voice detection settings
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
    
    # Speech contexts - These help Google STT recognize specific phrases better
    speech_contexts: list = Field(
        default=[
            {
                "phrases": ["price", "plan", "cost", "subscription", "service", "features", "support", 
                           "help", "agent", "assistant", "voice", "stop", "continue", "yes", "no"],
                "boost": 15.0
            }
        ],
        description="Speech contexts for improved recognition"
    )
    
    class Config:
        env_prefix = "STT_"
        case_sensitive = False

# Create a global config instance
config = STTConfig()