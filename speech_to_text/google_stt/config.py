"""
Configuration for Google Cloud Speech-to-Text.
"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class STTConfig(BaseSettings):
    """Configuration for Google Cloud STT."""
    
    # Google Cloud credentials
    google_credentials_path: str = Field(
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        description="Path to Google Cloud credentials JSON file"
    )
    
    # STT settings
    language_code: str = Field(
        default="en-US",
        description="Language code for recognition"
    )
    
    sample_rate: int = Field(
        default=8000,
        description="Audio sample rate in Hz - 8kHz for telephony"
    )
    
    model: str = Field(
        default="phone_call",
        description="Recognition model (phone_call, video, default, etc.)"
    )
    
    use_enhanced: bool = Field(
        default=True,
        description="Whether to use enhanced model for better accuracy"
    )
    
    enable_automatic_punctuation: bool = Field(
        default=True,
        description="Whether to add punctuation to transcriptions"
    )
    
    # Streaming settings
    interim_results: bool = Field(
        default=True,
        description="Whether to return interim results"
    )
    
    # Speech contexts for better recognition
    speech_contexts: list = Field(
        default=["help", "price", "plan", "cost", "support", "agent", "stop", "repeat"],
        description="Phrases to boost in recognition"
    )
    
    speech_context_boost: float = Field(
        default=10.0,
        description="Boost factor for speech contexts (0.0 to 20.0)"
    )
    
    # Barge-in settings
    enable_barge_in: bool = Field(
        default=True,
        description="Whether to enable barge-in detection"
    )
    
    barge_in_threshold: float = Field(
        default=0.045,
        description="Energy threshold for barge-in detection"
    )
    
    barge_in_min_duration_ms: int = Field(
        default=300,
        description="Minimum duration of speech for barge-in"
    )
    
    barge_in_cooldown_ms: int = Field(
        default=1500,
        description="Cooldown period after agent starts speaking"
    )
    
    # Buffer settings
    min_buffer_size: int = Field(
        default=1600,
        description="Minimum buffer size for processing (bytes)"
    )
    
    max_buffer_size: int = Field(
        default=8000,
        description="Maximum buffer size before processing (bytes)"
    )
    
    # Other settings
    enable_profanity_filter: bool = Field(
        default=False,
        description="Whether to filter profanity"
    )
    
    class Config:
        env_prefix = "STT_"
        case_sensitive = False

# Create global config instance
config = STTConfig()