"""
Generalized Deepgram STT implementation for better transcription quality
with any type of conversation.
"""
import os
import logging
import asyncio
import aiohttp
import json
import base64
from typing import Dict, Any, Optional, Union, List, AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass

from ..config import config
from .models import TranscriptionResult, TranscriptionConfig
from .exceptions import STTError, STTAPIError, STTStreamingError

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None

class DeepgramStreamingSTT:
    """
    Generalized version of DeepgramStreamingSTT that uses REST API
    with enhanced parameters for good transcription quality in any context.
    """
    
    API_URL = "https://api.deepgram.com/v1/listen"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        channels: int = 1,
        interim_results: Optional[bool] = None
    ):
        """
        Initialize the Deepgram STT client.
        
        Args:
            api_key: Deepgram API key (defaults to environment variable)
            model_name: STT model to use (defaults to config)
            language: Language code for recognition (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            encoding: Audio encoding format (default linear16)
            channels: Number of audio channels (default 1)
            interim_results: Whether to return interim results (default True)
        """
        self.api_key = api_key or config.deepgram_api_key
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Set it in .env file or pass directly.")
        
        self.model_name = model_name or "general"  # Hardcoded to "general" model which is available
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results if interim_results is not None else config.interim_results
        
        # State management for simulating streaming
        self.is_streaming = False
        self.session = None
        self.chunk_buffer = bytearray()
        self.buffer_size_threshold = 16384  # Process when buffer reaches ~16KB for better context
        self.utterance_id = 0
        self.last_result = None
    
    def _get_params(self) -> Dict[str, Any]:
        """Get optimized parameters for the API request."""
        params = {
            "model": self.model_name,
            "language": self.language,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "utterance_end_ms": str(config.utterance_end_ms),
            "smart_format": str(config.smart_format).lower(),
            "filler_words": "false",  # Filter out filler words
            "profanity_filter": str(config.profanity_filter).lower(),
            "alternatives": str(config.alternatives),
            "tier": config.model_options.get("tier", "enhanced"),
            "punctuate": "true",  # Add punctuation for better readability
            "diarize": "false"     # Single speaker for telephony
        }
        
        return params
    
    async def start_streaming(
        self, 
        config_obj: Optional[TranscriptionConfig] = None
    ) -> None:
        """
        Simulate starting a streaming session.
        
        Args:
            config_obj: Optional configuration object
        """
        if self.is_streaming:
            await self.stop_streaming()
        
        # Create a new aiohttp session
        self.session = aiohttp.ClientSession()
        
        # Reset buffer
        self.chunk_buffer = bytearray()
        self.utterance_id = 0
        self.is_streaming = True
        
        logger.info("Started Deepgram session (simulated streaming)")
    
    async def stop_streaming(self) -> None:
        """Stop the simulated streaming session."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Process any remaining audio in the buffer
        if len(self.chunk_buffer) > 0:
            try:
                await self._process_buffer()
            except Exception as e:
                logger.error(f"Error processing final buffer: {e}")
        
        # Close the session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Stopped Deepgram session")
    
    async def process_audio_chunk(
        self, 
        audio_chunk: Union[bytes, bytearray, memoryview],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process an audio chunk.
        
        Args:
            audio_chunk: Audio data chunk
            callback: Optional async callback for results
            
        Returns:
            Optional transcription result
        """
        if not self.is_streaming or not self.session:
            raise STTStreamingError("Not streaming - call start_streaming() first")
        
        try:
            # Ensure audio is in bytes format
            if not isinstance(audio_chunk, (bytes, bytearray, memoryview)):
                raise ValueError("Audio chunk must be bytes, bytearray, or memoryview")
            
            # Add to buffer
            self.chunk_buffer.extend(audio_chunk)
            
            # Process buffer if it's large enough
            if len(self.chunk_buffer) >= self.buffer_size_threshold:
                return await self._process_buffer(callback)
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            raise STTStreamingError(f"Error processing audio chunk: {e}")
    
    async def _process_buffer(
        self, 
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process the current audio buffer with optimized parameters.
        
        Args:
            callback: Optional async callback for results
            
        Returns:
            Optional transcription result
        """
        if len(self.chunk_buffer) == 0:
            return None
            
        # Create a copy of the buffer
        audio_data = bytes(self.chunk_buffer)
        
        # Clear the buffer
        self.chunk_buffer = bytearray()
        
        # Get optimized parameters
        params = self._get_params()
        
        # Create headers
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/raw"
        }
        
        try:
            # Send request to Deepgram API
            async with self.session.post(
                self.API_URL,
                params=params,
                headers=headers,
                data=audio_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Deepgram API error: {response.status}, {error_text}")
                    return None
                
                # Parse the response
                response_data = await response.json()
                
                # Log the full response for debugging
                logger.debug(f"Deepgram response: {json.dumps(response_data, indent=2)}")
                
                # Process the results
                results = response_data.get("results", {})
                channels = results.get("channels", [{}])
                
                # Get the best channel (usually only one)
                channel = channels[0] if channels else {}
                alternatives = channel.get("alternatives", [{}])
                
                # Get the best alternative
                if alternatives:
                    alternative = alternatives[0]
                    transcript = alternative.get("transcript", "")
                    confidence = alternative.get("confidence", 0.0)
                    words = alternative.get("words", [])
                    
                    # Create a result object
                    self.utterance_id += 1
                    result = StreamingTranscriptionResult(
                        text=transcript,
                        is_final=True,  # Always final in this implementation
                        confidence=confidence,
                        words=words,
                        chunk_id=self.utterance_id
                    )
                    
                    # Store result
                    self.last_result = result
                    
                    # Call callback if provided
                    if callback and transcript.strip():
                        await callback(result)
                    
                    return result
                
                return None
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
            return None
    
    async def stream_audio_file(
        self,
        file_path: str,
        chunk_size: int = 4096,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None,
        simulate_realtime: bool = False
    ) -> List[StreamingTranscriptionResult]:
        """
        Stream an audio file to Deepgram.
        
        Args:
            file_path: Path to audio file
            chunk_size: Size of audio chunks to send
            callback: Optional async callback for results
            simulate_realtime: Whether to simulate real-time streaming
            
        Returns:
            List of final transcription results
        """
        try:
            # Start streaming
            await self.start_streaming()
            
            # Open the audio file
            with open(file_path, 'rb') as f:
                # Collect final results
                final_results = []
                
                # Stream chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Process chunk
                    result = await self.process_audio_chunk(chunk, callback)
                    
                    # Add final result if available
                    if result:
                        final_results.append(result)
                    
                    # Simulate real-time streaming if requested
                    if simulate_realtime:
                        await asyncio.sleep(chunk_size / self.sample_rate / 2)  # Half real-time speed
            
            # Close the stream
            await self.stop_streaming()
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error streaming audio file: {e}")
            await self.stop_streaming()
            raise STTStreamingError(f"Error streaming audio file: {str(e)}")