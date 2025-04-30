"""
Configuration settings for the Text-to-Speech module using Google Cloud TTS.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class TTSConfig(BaseSettings):
    """Configuration for Text-to-Speech module using Google Cloud TTS."""
    
    # Google Cloud TTS settings
    google_application_credentials: str = Field(
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        description="Path to Google Cloud credentials JSON file"
    )
    
    # TTS settings - Using Standard voice instead of Neural to avoid SSML issues
    voice_name: str = Field(
        default="en-US-Standard-D",
        description="Google TTS voice name to use"
    )
    
    voice_gender: str = Field(
        default="MALE",
        description="Voice gender (MALE/FEMALE/NEUTRAL)"
    )
    
    language_code: str = Field(
        default="en-US",
        description="Language code for TTS"
    )
    
    # Telephony-optimized settings
    sample_rate: int = Field(
        default=8000,  # Changed to 8kHz for telephony
        description="Audio sample rate in Hz - 8kHz is best for telephony"
    )
    
    audio_encoding: str = Field(
        default="LINEAR16",
        description="Audio encoding format (LINEAR16, MP3, etc.)"
    )
    
    audio_profile: str = Field(
        default="telephony-class-application",
        description="Audio profile for optimization"
    )
    
    # Voice quality settings
    speaking_rate: float = Field(
        default=1.1,  # Slightly faster than normal, but not too fast
        description="Speaking rate in API (1.0 is normal, >1.0 is faster)"
    )
    
    pitch: float = Field(
        default=0.0,
        description="Pitch adjustment in semitones (0.0 is normal)"
    )
    
    # SSML settings for extra control
    ssml_rate: str = Field(
        default="1.1",  # Slightly faster than normal
        description="SSML speaking rate value (can be numeric like '1.1' or text like 'medium')"
    )
    
    ssml_pitch: str = Field(
        default="0",
        description="SSML pitch adjustment in semitones (e.g. '0', '+2', '-1')"
    )
    
    # Explicitly enable SSML by default
    use_ssml: bool = Field(
        default=True,
        description="Whether to use SSML for all TTS requests"
    )
    
    # Streaming settings
    chunk_size: int = Field(
        default=1024,
        description="Size of audio chunks to process at once in bytes"
    )
    
    max_text_chunk_size: int = Field(
        default=250,
        description="Maximum text chunk size to send to Google Cloud at once"
    )
    
    stream_timeout: float = Field(
        default=10.0,
        description="Timeout for streaming operations in seconds"
    )
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of TTS results"
    )
    
    cache_dir: str = Field(
        default="./cache/tts_cache",
        description="Directory for caching TTS results"
    )
    
    # Fallback settings
    fallback_message: str = Field(
        default="I'm sorry, I'm having trouble generating speech at the moment.",
        description="Fallback message when TTS fails"
    )
    
    # SSML telephony optimizations
    enable_telephony_optimization: bool = Field(
        default=True,
        description="Enable SSML optimizations for telephony"
    )
    
    # Using a simple SSML template without emphasis tags which can cause issues
    telephony_ssml_template: str = Field(
        default='<speak><prosody rate="{rate}">{text}</prosody></speak>',
        description="SSML template for telephony optimization"
    )
    
    class Config:
        env_prefix = "TTS_"
        case_sensitive = False

# Create a global config instance
config = TTSConfig()