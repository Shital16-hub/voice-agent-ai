"""
Deepgram Text-to-Speech client for the Voice AI Agent.
"""
import os
import logging
import asyncio
import aiohttp
import hashlib
import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, Any, Union

from .config import config
from .exceptions import TTSError

logger = logging.getLogger(__name__)

class DeepgramTTS:
    """
    Client for the Deepgram Text-to-Speech API with support for streaming.
    
    This class handles both batch and streaming TTS operations using Deepgram's API,
    optimized for low-latency voice AI applications.
    """

    BASE_URL = "https://api.deepgram.com/v1/speak"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        sample_rate: Optional[int] = None,
        container_format: Optional[str] = None,
        enable_caching: Optional[bool] = None
    ):
        """
        Initialize the Deepgram TTS client.
        
        Args:
            api_key: Deepgram API key (defaults to environment variable)
            model: TTS model to use (defaults to config)
            voice: Voice for synthesis (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            container_format: Audio format (defaults to config)
            enable_caching: Whether to cache results (defaults to config)
        """
        self.api_key = api_key or config.deepgram_api_key
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Set it in .env file or pass directly.")
        
        self.model = model or config.model
        self.voice = voice or config.voice
        self.sample_rate = sample_rate or config.sample_rate
        self.container_format = container_format or config.container_format
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for Deepgram API requests."""
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_params(self, **kwargs) -> Dict[str, Any]:
        """Get the parameters for a TTS request, with overrides from kwargs."""
        params = {
            "model": self.model,
            "voice": self.voice,
        }
        
        # Only add sample_rate if not using MP3 encoding
        if self.container_format != "mp3":
            params["sample_rate"] = self.sample_rate
        
        # Add encoding/container format
        if self.container_format:
            params["encoding"] = self.container_format
        
        # Add optional parameters if provided
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value
                
        return params
    
    def _get_cache_path(self, text: str, params: Dict[str, Any]) -> Path:
        """
        Generate a cache file path based on text and parameters.
        
        Args:
            text: Text to synthesize
            params: TTS parameters
            
        Returns:
            Path to the cache file
        """
        # Create a unique hash based on text and params
        cache_key = hashlib.md5(f"{text}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.{self.container_format}"
    
    async def synthesize(
        self, 
        text: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech in a single request.
        
        Args:
            text: Text to synthesize
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Audio data as bytes
        """
        params = self._get_params(**kwargs)
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(text, params)
            if cache_path.exists():
                logger.debug(f"Found cached TTS result for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Prepare request
        payload = {
            "text": text
        }
        
        # Add optional parameters as query parameters
        query_params = {}
        for key, value in params.items():
            if key != "text" and value is not None:
                query_params[key] = value
        
        # Make API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=self._get_headers(),
                    json=payload,
                    params=query_params
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise TTSError(f"Deepgram API error: {response.status} - {error_msg}")
                    
                    audio_data = await response.read()
            
            # Cache result if enabled
            if self.enable_caching:
                cache_path.write_bytes(audio_data)
                
            return audio_data
            
        except aiohttp.ClientError as e:
            raise TTSError(f"Network error when connecting to Deepgram: {str(e)}")
        except Exception as e:
            raise TTSError(f"Error during TTS synthesis: {str(e)}")
    
    async def synthesize_streaming(
        self, 
        text_stream: AsyncGenerator[str, None],
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech synthesis for real-time applications.
        
        Takes a streaming text input and returns streaming audio output,
        optimized for low-latency voice applications.
        
        Args:
            text_stream: Async generator producing text chunks
            **kwargs: Additional parameters to pass to the API
            
        Yields:
            Audio data chunks as they are generated
        """
        buffer = ""
        max_chunk_size = config.max_text_chunk_size
        
        try:
            async for text_chunk in text_stream:
                if not text_chunk:
                    continue
                    
                # Add to buffer
                buffer += text_chunk
                
                # Process buffer if it's large enough or contains sentence-ending punctuation
                if len(buffer) >= max_chunk_size or any(c in buffer for c in ['.', '!', '?', '\n']):
                    # Process the buffered text
                    audio_data = await self.synthesize(buffer, **kwargs)
                    yield audio_data
                    buffer = ""
            
            # Process any remaining text in the buffer
            if buffer:
                audio_data = await self.synthesize(buffer, **kwargs)
                yield audio_data
                
        except Exception as e:
            logger.error(f"Error in streaming TTS: {str(e)}")
            raise TTSError(f"Streaming TTS error: {str(e)}")
    
    async def synthesize_with_ssml(
        self, 
        ssml: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize speech using SSML markup for advanced control.
        
        Args:
            ssml: SSML-formatted text
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Audio data as bytes
        """
        # Ensure SSML is properly formatted
        if not ssml.startswith('<speak>'):
            ssml = f"<speak>{ssml}</speak>"
            
        # Create a payload with just the SSML text
        payload = {
            "text": ssml
        }
        
        # Add SSML flag to query parameters
        kwargs['ssml'] = True
        
        # Get parameters
        params = self._get_params(**kwargs)
        
        # Create query parameters
        query_params = {}
        for key, value in params.items():
            if key != "text" and value is not None:
                query_params[key] = value
        
        # Make API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=self._get_headers(),
                    json=payload,
                    params=query_params
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise TTSError(f"Deepgram API error: {response.status} - {error_msg}")
                    
                    audio_data = await response.read()
            
            # Cache result if enabled
            if self.enable_caching:
                cache_path = self._get_cache_path(ssml, params)
                cache_path.write_bytes(audio_data)
                
            return audio_data
            
        except aiohttp.ClientError as e:
            raise TTSError(f"Network error when connecting to Deepgram: {str(e)}")
        except Exception as e:
            raise TTSError(f"Error during TTS synthesis: {str(e)}")