# speech_to_text/config.py
"""
Configuration settings for the Google Cloud Speech integration module.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class STTConfig(BaseSettings):
    """Configuration for Speech-to-Text module."""
    
    # Google Cloud Speech settings
    google_credentials_path: str = Field(
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        description="Path to Google Cloud credentials JSON file"
    )
    
    # STT settings
    model: str = Field(
        default="phone_call",
        description="Google Speech-to-Text model to use"
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
    
    enhanced: bool = Field(
        default=True,
        description="Whether to use enhanced models"
    )
    
    # Telephony optimizations
    keywords: list = Field(
        default=["price", "plan", "cost", "subscription", "service", "features", "support"],
        description="Keywords to boost in telephony context"
    )
    
    # Performance Settings - Optimized for noise handling
    enable_noise_filtering: bool = Field(
        default=True,
        description="Enable enhanced noise filtering"
    )
    
    silence_threshold: float = Field(
        default=0.008,
        description="Threshold for silence detection"
    )
    
    silence_duration: float = Field(
        default=1.2,
        description="Duration of silence to consider an utterance complete"
    )
    
    # Response Settings
    min_transcription_length: int = Field(
        default=3,
        description="Minimum word count for valid transcription"
    )
    
    # Caching settings
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of STT results"
    )
    
    class Config:
        env_prefix = "STT_"
        case_sensitive = False

# Create a global config instance
config = STTConfig()