"""
Configuration settings for the Text-to-Speech module.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class TTSConfig(BaseSettings):
    """Configuration for Text-to-Speech module."""
    
    # Deepgram API settings
    deepgram_api_key: str = Field(
        default=os.getenv("DEEPGRAM_API_KEY", ""),
        description="Deepgram API Key for TTS services"
    )
    
    # TTS settings
    model: str = Field(
        default="aura-asteria-en",
        description="Deepgram TTS model to use"
    )
    voice: str = Field(
        default="aura-asteria-en",  # Default voice
        description="Voice for the TTS system"
    )
    sample_rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz"
    )
    container_format: str = Field(
        default="mp3",
        description="Audio container format (mp3, wav, etc.)"
    )
    
    # Streaming settings
    chunk_size: int = Field(
        default=1024,
        description="Size of audio chunks to process at once in bytes"
    )
    max_text_chunk_size: int = Field(
        default=100,
        description="Maximum text chunk size to send to Deepgram at once"
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
    
    class Config:
        env_prefix = "TTS_"
        case_sensitive = False

# Create a global config instance
config = TTSConfig()