"""
This is an updated version of DeepgramStreamingSTT to fix the streaming state management issues.
Replace the streaming.py file in your speech_to_text/deepgram_stt/ directory with this content.
"""

"""
Generalized Deepgram STT implementation for better transcription quality
with any type of conversation, with enhanced telephony optimization.
"""
import os
import logging
import asyncio
import aiohttp
import json
import base64
import time
from typing import Dict, Any, Optional, Union, List, AsyncGenerator, Callable, Awaitable, Tuple
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
    with enhanced parameters for superior transcription quality in telephony contexts.
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
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Set it in .env file or pass directly.")
        
        self.model_name = model_name or "general"  # Use general model instead of enhanced
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results if interim_results is not None else config.interim_results
        
        # State management for simulating streaming
        self.is_streaming = False
        self.session = None
        self.chunk_buffer = bytearray()
        self.buffer_size_threshold = 12288  # Reduced from 16384 for lower latency (12KB)
        self.utterance_id = 0
        self.last_result = None
        
        # Add a lock to prevent concurrent streaming operations
        self.streaming_lock = asyncio.Lock()
        
        # Create a start completion event to track when streaming is truly started
        self.streaming_started = asyncio.Event()
        self.streaming_started.clear()
        
        # Telephony-optimized parameters
        self.smart_format = True  # Enable smart formatting for better readability
        self.endpointing = 700  # 700ms for better segmentation (increased from default)
        self.vad_turnoff = 650  # 650ms VAD turnoff for better phrase detection
        
        # Noise reduction
        self.diarize = False  # Disable diarization - single speaker in telephony
        self.multichannel = False  # Single channel for telephony
        self.filler_words = False  # Disable filler words for cleaner transcription
        
        # Boost important keywords for telephony context
        self.keywords = [
            "help", "agent", "person", "service", "support", 
            "customer", "assistance", "problem", "issue", "question",
            "price", "cost", "plan", "subscription", "payment"
        ]
        
        logger.info(f"Initialized DeepgramStreamingSTT with model: {self.model_name}")
    
    def _get_params(self) -> Dict[str, Any]:
        """Get optimized parameters for the API request."""
        params = {
            "model": self.model_name,
            "language": self.language,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "utterance_end_ms": "700",  # Increased for better chunking
            "vad_turnoff": "650",  # Increased for better silence detection
            "smart_format": "true",  # Enable smart formatting
            "filler_words": "false",  # Filter out filler words
            "profanity_filter": "false",  # Don't filter profanity for telephony
            "alternatives": "1",  # Just one alternative for speed
            "tier": "base",  # Use base tier instead of enhanced to avoid permission issues
            "punctuate": "true",  # Add punctuation
            "diarize": "false",   # Single speaker for telephony
            "numerals": "true",   # Convert numbers to numerals
            "interim_results": "true" if self.interim_results else "false"
        }
        
        # Add keywords if available
        if self.keywords:
            params["keywords"] = json.dumps(self.keywords)
        
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
        # Only lock for starting/stopping operations
        async with self.streaming_lock:
            if self.is_streaming:
                await self.stop_streaming()
            
            # Create a new aiohttp session with optimized timeout
            timeout = aiohttp.ClientTimeout(total=30, connect=3, sock_connect=3, sock_read=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Reset buffer
            self.chunk_buffer = bytearray()
            self.utterance_id = 0
            self.is_streaming = True
            
            # Signal that streaming has started
            self.streaming_started.set()
            
            logger.info("Started Deepgram session with optimized parameters")
    
    async def stop_streaming(self) -> Optional[Tuple[str, float]]:
        """
        Stop the simulated streaming session.
        
        Returns:
            Optional tuple of (transcription, duration) or None on error
        """
        # Only lock for starting/stopping operations
        async with self.streaming_lock:
            if not self.is_streaming:
                logger.warning("stop_streaming called but not currently streaming")
                return None
            
            self.is_streaming = False
            self.streaming_started.clear()
            
            # Process any remaining audio in the buffer
            result = None
            if len(self.chunk_buffer) > 0:
                try:
                    result = await self._process_buffer()
                except Exception as e:
                    logger.error(f"Error processing final buffer: {e}")
            
            # Close the session
            if self.session:
                await self.session.close()
                self.session = None
            
            logger.info("Stopped Deepgram session")
            
            # Return transcription result if available
            if result and hasattr(result, 'text'):
                return result.text, 0.0
            
            return None
    
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
        # No lock here to avoid deadlocks
        if not self.is_streaming or not self.session:
            # Check if streaming is initialized
            if not self.is_streaming:
                # Auto-initialize streaming if needed
                try:
                    await self.start_streaming()
                    # Wait for streaming to be properly initialized
                    await asyncio.wait_for(self.streaming_started.wait(), timeout=5.0)
                except Exception as e:
                    raise STTStreamingError(f"Failed to auto-initialize streaming: {e}")
            else:
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
        
        # Verify we're still streaming
        if not self.is_streaming or not self.session:
            logger.warning("Not streaming while processing buffer")
            return None
        
        # Get optimized parameters
        params = self._get_params()
        
        # Create headers
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/raw"
        }
        
        try:
            # Send request to Deepgram API with optimized timeout
            async with self.session.post(
                self.API_URL,
                params=params,
                headers=headers,
                data=audio_data,
                timeout=5.0  # Reduced timeout for better responsiveness
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Deepgram API error: {response.status}, {error_text}")
                    return None
                
                # Parse the response
                response_data = await response.json()
                
                # Log the response structure for debugging
                if logger.isEnabledFor(logging.DEBUG):
                    debug_response = {
                        "has_results": "results" in response_data,
                        "channels": len(response_data.get("results", {}).get("channels", [])) if "results" in response_data else 0
                    }
                    logger.debug(f"Deepgram response structure: {debug_response}")
                
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
            # Wait for streaming to be properly initialized
            await asyncio.wait_for(self.streaming_started.wait(), timeout=5.0)
            
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