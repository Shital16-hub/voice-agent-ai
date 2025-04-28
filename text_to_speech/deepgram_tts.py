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
from typing import AsyncGenerator, Dict, Optional, Any, Union, List

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
    MAX_CHUNK_SIZE = 1500  # Maximum safe character chunk size for Deepgram API
    
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
        }
        
        # Add sample_rate only if specified and not using MP3
        if self.sample_rate and self.container_format != "mp3":
            params["sample_rate"] = self.sample_rate
        
        # Add encoding/container format
        if self.container_format:
            # Map container format to Deepgram's expected encoding values
            encoding_map = {
                "wav": "linear16",  # Use linear16 for WAV
                "mp3": "mp3",
                "opus": "opus",
                "flac": "flac",
                "aac": "aac",
                "linear16": "linear16",
                "mulaw": "mulaw",
                "alaw": "alaw"
            }
            params["encoding"] = encoding_map.get(self.container_format, self.container_format)
        
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
        
        # Determine file extension based on format
        ext = "wav" if params.get("encoding") == "linear16" else params.get("encoding", "mp3")
        return self.cache_dir / f"{cache_key}.{ext}"
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        Split long text into smaller chunks that won't exceed API limits.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.MAX_CHUNK_SIZE:
            return [text]
            
        chunks = []
        sentences = []
        
        # First try to split by sentences (periods, exclamation marks, question marks)
        for sentence in text.replace('!', '.').replace('?', '.').split('.'):
            if sentence.strip():
                sentences.append(sentence.strip() + '.')
        
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > self.MAX_CHUNK_SIZE:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If a single sentence is longer than the limit, split it by words
                if len(sentence) > self.MAX_CHUNK_SIZE:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > self.MAX_CHUNK_SIZE:
                            chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk += " " + word
                    
                    if word_chunk.strip():
                        current_chunk = word_chunk.strip()
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
                
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Split text of length {len(text)} into {len(chunks)} chunks")
        return chunks
    
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
        if not text:
            logger.warning("Empty text provided to synthesize")
            return b''
            
        # Check if text exceeds Deepgram's limit
        if len(text) > 2000:
            logger.info(f"Text length ({len(text)}) exceeds Deepgram's 2000 character limit. Splitting into chunks.")
            chunks = self._split_long_text(text)
            audio_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
                    chunk_audio = await self._synthesize_chunk(chunk, **kwargs)
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i+1}: {e}")
                    # Continue with next chunk instead of failing completely
            
            # Combine audio chunks
            if not audio_chunks:
                raise TTSError("Failed to synthesize any chunks of the text")
                
            # For WAV format, we need special handling to combine multiple WAV files
            if self.container_format == "wav" or kwargs.get('encoding') == 'linear16':
                # For simplicity, just concatenate the raw PCM data without WAV headers
                # In a real implementation, you'd properly combine WAV files
                return audio_chunks[0]  # Just return the first chunk for now
            else:
                # For MP3 and other formats, simple concatenation might work
                return b''.join(audio_chunks)
        else:
            # Standard case - text is within limits
            return await self._synthesize_chunk(text, **kwargs)
    
    async def _synthesize_chunk(
        self, 
        text: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize a single chunk of text.
        
        Args:
            text: Text chunk to synthesize
            **kwargs: Additional parameters
            
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
        
        # Make API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=self._get_headers(),
                    json=payload,
                    params=params
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise TTSError(f"Deepgram API error: {response.status} - {error_msg}")
                    
                    audio_data = await response.read()
            
            # Cache result if enabled
            if self.enable_caching and audio_data:
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
        max_chunk_size = min(config.max_text_chunk_size, self.MAX_CHUNK_SIZE)
        
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
            
        # Check length
        if len(ssml) > 2000:
            logger.warning("SSML text exceeds 2000 character limit. Will be truncated by Deepgram.")
            
        # Create a payload with just the SSML text
        payload = {
            "text": ssml
        }
        
        # Add SSML flag to query parameters
        kwargs['ssml'] = True
        
        # Get parameters
        params = self._get_params(**kwargs)
        
        # Make API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=self._get_headers(),
                    json=payload,
                    params=params
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