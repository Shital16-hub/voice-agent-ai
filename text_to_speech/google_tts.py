"""
Google Cloud Text-to-Speech client for the Voice AI Agent.
"""
import os
import logging
import asyncio
import hashlib
import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, Any, Union, List, Callable, Awaitable
import io

# Google Cloud TTS
from google.cloud import texttospeech
from google.oauth2 import service_account

from .config import config
from .exceptions import TTSError, TTSAPIError

logger = logging.getLogger(__name__)

class GoogleCloudTTS:
    """
    Client for the Google Cloud Text-to-Speech API with support for streaming.
    
    This class handles both batch and streaming TTS operations using Google Cloud's API,
    optimized for low-latency voice AI applications.
    """
    
    MAX_CHAR_COUNT = 5000  # Google Cloud TTS character limit
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None,
        voice_name: Optional[str] = None,
        language_code: Optional[str] = None,
        audio_encoding: Optional[str] = None,
        audio_profile: Optional[str] = None,
        enable_caching: Optional[bool] = None
    ):
        """
        Initialize the Google Cloud TTS client.
        
        Args:
            credentials_path: Path to Google credentials JSON (defaults to env variable)
            voice_name: Voice name to use (defaults to config)
            language_code: Language code (defaults to config)
            audio_encoding: Audio encoding format (defaults to config)
            audio_profile: Audio profile for optimization (defaults to config)
            enable_caching: Whether to cache results (defaults to config)
        """
        self.credentials_path = credentials_path or config.google_application_credentials
        if not self.credentials_path:
            raise ValueError("Google Cloud credentials path is required. Set GOOGLE_APPLICATION_CREDENTIALS in .env or pass directly.")
            
        self.voice_name = voice_name or config.voice_name
        self.language_code = language_code or config.language_code
        self.audio_encoding = audio_encoding or config.audio_encoding
        self.audio_profile = audio_profile or config.audio_profile
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the Google Cloud TTS client with the provided credentials."""
        try:
            # Check if credentials file exists
            if not os.path.exists(self.credentials_path):
                raise ValueError(f"Google Cloud credentials file not found: {self.credentials_path}")
                
            # Initialize credentials
            credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
            
            # Create the client
            self.client = texttospeech.TextToSpeechClient(credentials=credentials)
            logger.info(f"Initialized Google Cloud TTS client with voice: {self.voice_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud TTS client: {e}")
            raise TTSAPIError(f"Failed to initialize Google Cloud TTS client: {str(e)}")
    
    def _get_voice_params(self) -> Dict[str, Any]:
        """Get voice parameters for synthesis request."""
        # Parse the voice name to extract language and voice type
        voice_parts = self.voice_name.split('-')
        
        # Extract gender from config or fallback to NEUTRAL
        gender = getattr(texttospeech.SsmlVoiceGender, config.voice_gender)
        
        return {
            "language_code": self.language_code,
            "name": self.voice_name,
            "ssml_gender": gender
        }
    
    def _get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration for synthesis request."""
        # Map string encoding to enum
        encoding_map = {
            "LINEAR16": texttospeech.AudioEncoding.LINEAR16,
            "MP3": texttospeech.AudioEncoding.MP3,
            "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS,
            "MULAW": texttospeech.AudioEncoding.MULAW
        }
        
        encoding = encoding_map.get(self.audio_encoding, texttospeech.AudioEncoding.LINEAR16)
        
        # Create audio config
        audio_config = {
            "audio_encoding": encoding,
            "sample_rate_hertz": config.sample_rate,
        }
        
        # Add effects profile if specified
        if self.audio_profile:
            audio_config["effects_profile_id"] = [self.audio_profile]
            
        return audio_config
    
    def _get_cache_path(self, text: str, is_ssml: bool) -> Path:
        """
        Generate a cache file path based on text and parameters.
        
        Args:
            text: Text to synthesize
            is_ssml: Whether the text is SSML
            
        Returns:
            Path to the cache file
        """
        # Create a unique hash based on text and parameters
        params = {
            "voice_name": self.voice_name,
            "language_code": self.language_code,
            "audio_encoding": self.audio_encoding,
            "audio_profile": self.audio_profile,
            "is_ssml": is_ssml
        }
        
        cache_key = hashlib.md5(f"{text}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        
        # Determine file extension based on format
        ext = "wav" if self.audio_encoding == "LINEAR16" else "mp3"
        return self.cache_dir / f"{cache_key}.{ext}"
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        Split long text into smaller chunks that won't exceed API limits.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.MAX_CHAR_COUNT:
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
            if len(current_chunk) + len(sentence) > self.MAX_CHAR_COUNT:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If a single sentence is longer than the limit, split it by words
                if len(sentence) > self.MAX_CHAR_COUNT:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > self.MAX_CHAR_COUNT:
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
        is_ssml: bool = False
    ) -> bytes:
        """
        Synthesize text to speech in a single request.
        
        Args:
            text: Text to synthesize
            is_ssml: Whether the text is SSML
            
        Returns:
            Audio data as bytes
        """
        if not text:
            logger.warning("Empty text provided to synthesize")
            return b''
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(text, is_ssml)
            if cache_path.exists():
                logger.debug(f"Found cached TTS result for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Check if text exceeds Google Cloud's limit
        if len(text) > self.MAX_CHAR_COUNT:
            logger.info(f"Text length ({len(text)}) exceeds Google's {self.MAX_CHAR_COUNT} character limit. Splitting into chunks.")
            chunks = self._split_long_text(text)
            audio_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
                    chunk_audio = await self._synthesize_chunk(chunk, is_ssml)
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i+1}: {e}")
                    # Continue with next chunk instead of failing completely
            
            # Combine audio chunks
            if not audio_chunks:
                raise TTSError("Failed to synthesize any chunks of the text")
                
            # Concatenate audio data (this is a simplistic approach; may need refinement)
            combined_audio = b''.join(audio_chunks)
            
            # Cache the result if enabled
            if self.enable_caching:
                cache_path.write_bytes(combined_audio)
                
            return combined_audio
        else:
            # Standard case - text is within limits
            return await self._synthesize_chunk(text, is_ssml)
    
    async def _synthesize_chunk(
        self, 
        text: str,
        is_ssml: bool = False
    ) -> bytes:
        """
        Synthesize a single chunk of text using Google Cloud TTS.
        
        Args:
            text: Text chunk to synthesize
            is_ssml: Whether the text is SSML
            
        Returns:
            Audio data as bytes
        """
        try:
            # Run in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._synthesize_sync, text, is_ssml)
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {str(e)}")
            raise TTSError(f"Google Cloud TTS synthesis error: {str(e)}")
    
    def _synthesize_sync(self, text: str, is_ssml: bool) -> bytes:
        """
        Synchronous method to synthesize speech using Google Cloud TTS.
        This method is intended to be run in a separate thread.
        
        Args:
            text: Text to synthesize
            is_ssml: Whether the text is SSML
            
        Returns:
            Audio data as bytes
        """
        try:
            # Create the synthesis input
            synthesis_input = texttospeech.SynthesisInput()
            if is_ssml:
                # Ensure proper SSML format
                if not text.startswith('<speak>'):
                    text = f"<speak>{text}</speak>"
                synthesis_input.ssml = text
            else:
                synthesis_input.text = text
            
            # Configure voice
            voice_params = self._get_voice_params()
            voice = texttospeech.VoiceSelectionParams(**voice_params)
            
            # Configure audio
            audio_config_params = self._get_audio_config()
            audio_config = texttospeech.AudioConfig(**audio_config_params)
            
            # Perform synthesis
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Return the audio content
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Error in Google Cloud TTS synthesis: {e}")
            raise TTSError(f"Google Cloud TTS synthesis error: {str(e)}")
    
    async def synthesize_streaming(
        self, 
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech synthesis for real-time applications.
        
        Takes a streaming text input and returns streaming audio output,
        optimized for low-latency voice applications.
        
        Args:
            text_stream: Async generator producing text chunks
            
        Yields:
            Audio data chunks as they are generated
        """
        buffer = ""
        max_chunk_size = min(config.max_text_chunk_size, self.MAX_CHAR_COUNT)
        
        try:
            async for text_chunk in text_stream:
                if not text_chunk:
                    continue
                    
                # Add to buffer
                buffer += text_chunk
                
                # Process buffer if it's large enough or contains sentence-ending punctuation
                if len(buffer) >= max_chunk_size or any(c in buffer for c in ['.', '!', '?', '\n']):
                    # Process the buffered text
                    audio_data = await self.synthesize(buffer)
                    yield audio_data
                    buffer = ""
            
            # Process any remaining text in the buffer
            if buffer:
                audio_data = await self.synthesize(buffer)
                yield audio_data
                
        except Exception as e:
            logger.error(f"Error in streaming TTS: {str(e)}")
            raise TTSError(f"Streaming TTS error: {str(e)}")
    
    async def synthesize_with_ssml(
        self, 
        ssml: str
    ) -> bytes:
        """
        Synthesize speech using SSML markup for advanced control.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Audio data as bytes
        """
        return await self.synthesize(ssml, is_ssml=True)