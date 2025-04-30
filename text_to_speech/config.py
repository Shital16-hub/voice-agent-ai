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
    
    # TTS settings
    voice_name: str = Field(
        default="en-US-Neural2-F",
        description="Google TTS voice name to use"
    )
    
    voice_gender: str = Field(
        default="FEMALE",
        description="Voice gender (MALE/FEMALE/NEUTRAL)"
    )
    
    language_code: str = Field(
        default="en-US",
        description="Language code for TTS"
    )
    
    sample_rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz"
    )
    
    audio_encoding: str = Field(
        default="LINEAR16",
        description="Audio encoding format (LINEAR16, MP3, etc.)"
    )
    
    audio_profile: str = Field(
        default="telephony-class-application",
        description="Audio profile for optimization"
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
    
    # SSML settings
    enable_ssml: bool = Field(
        default=True,
        description="Enable SSML support"
    )
    
    # Fallback settings
    fallback_message: str = Field(
        default="I'm sorry, I'm having trouble generating speech at the moment.",
        description="Fallback message when TTS fails"
    )
    
    class Config:
        env_prefix = "TTS_"
        case_sensitive = False

# Create a global config instance
config = TTSConfig()