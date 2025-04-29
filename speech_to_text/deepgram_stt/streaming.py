"""
Deepgram Speech-to-Text streaming client for real-time transcription.
"""
import os
import logging
import asyncio
import aiohttp
import json
import base64
from typing import Dict, Any, Optional, Union, List, AsyncGenerator, Callable, Awaitable
import websockets
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
    Client for the Deepgram Speech-to-Text streaming API, optimized for telephony.
    
    This class handles real-time streaming STT operations using Deepgram's
    WebSocket API, with telephony-specific optimizations.
    """
    
    WEBSOCKET_URL = "wss://api.deepgram.com/v1/listen"
    
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
        Initialize the Deepgram streaming STT client.
        
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
        
        self.model_name = model_name or config.model_name
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results if interim_results is not None else config.interim_results
        
        # State management
        self.websocket = None
        self.is_streaming = False
        self.stream_task = None
        self.utterance_id = 0
        self.last_result = None
    
    def _get_ws_url_with_params(self, config_obj: Optional[TranscriptionConfig] = None) -> str:
        """
        Get WebSocket URL with parameters.
        
        Args:
            config_obj: Optional configuration object
            
        Returns:
            WebSocket URL with parameters
        """
        # Start with base parameters
        params = {
            "model": self.model_name,
            "language": self.language,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "interim_results": str(self.interim_results).lower(),
            "endpointing": config.endpointing,
            "vad_events": str(config.vad_events).lower(),
            "utterance_end_ms": str(config.utterance_end_ms),
            "smart_format": str(config.smart_format).lower(),
            "filler_words": "false",  # Filter out filler words
            "profanity_filter": str(config.profanity_filter).lower(),
            "alternatives": str(config.alternatives),
            "tier": config.model_options.get("tier", "enhanced"),
        }
        
        # Add any parameters from config_obj
        if config_obj:
            config_dict = config_obj.dict(exclude_none=True, exclude_unset=True)
            params.update(config_dict)
        
        # Build query string
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        
        return f"{self.WEBSOCKET_URL}?{query_string}"
    
    async def start_streaming(
        self, 
        config_obj: Optional[TranscriptionConfig] = None
    ) -> None:
        """
        Start a streaming session.
        
        Args:
            config_obj: Optional configuration object
        """
        if self.is_streaming:
            await self.stop_streaming()
        
        # Get WebSocket URL with parameters
        ws_url = self._get_ws_url_with_params(config_obj)
        
        # Connect to WebSocket
        try:
            self.websocket = await websockets.connect(
                ws_url,
                extra_headers={"Authorization": f"Token {self.api_key}"},
                ping_interval=5,
                ping_timeout=10
            )
            
            self.is_streaming = True
            self.utterance_id = 0
            logger.info("Started Deepgram streaming session")
            
        except websockets.exceptions.WebSocketException as e:
            self.is_streaming = False
            raise STTStreamingError(f"Failed to connect to Deepgram WebSocket: {str(e)}")
        except Exception as e:
            self.is_streaming = False
            raise STTStreamingError(f"Error starting streaming session: {str(e)}")
    
    async def stop_streaming(self) -> None:
        """Stop the streaming session."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Close WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {str(e)}")
            finally:
                self.websocket = None
        
        # Cancel stream task if running
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
            self.stream_task = None
        
        logger.info("Stopped Deepgram streaming session")
    
    async def process_audio_chunk(
        self, 
        audio_chunk: Union[bytes, bytearray, memoryview],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process an audio chunk and get transcription.
        
        Args:
            audio_chunk: Audio data chunk
            callback: Optional async callback for interim results
            
        Returns:
            Optional transcription result (if a final result is available)
        """
        if not self.is_streaming or not self.websocket:
            raise STTStreamingError("Not streaming - call start_streaming() first")
        
        try:
            # Ensure audio is in bytes format
            if not isinstance(audio_chunk, (bytes, bytearray, memoryview)):
                raise ValueError("Audio chunk must be bytes, bytearray, or memoryview")
            
            # Send audio chunk to Deepgram
            await self.websocket.send(audio_chunk)
            
            # Process any messages that are ready
            result = await self._process_messages(callback)
            return result
            
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {str(e)}")
            self.is_streaming = False
            raise STTStreamingError(f"WebSocket error: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            raise STTStreamingError(f"Error processing audio chunk: {str(e)}")
    
    async def _process_messages(
        self, 
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process incoming messages from Deepgram.
        
        Args:
            callback: Optional async callback for results
            
        Returns:
            Optional transcription result (if a final result is available)
        """
        # Check for available messages without blocking
        try:
            # Try to receive a message with a short timeout
            response_text = await asyncio.wait_for(self.websocket.recv(), timeout=0.01)
            
            # Parse the response
            response = json.loads(response_text)
            
            # Check for speech detected event
            if "type" in response and response["type"] == "SpeechStarted":
                logger.debug("Speech started detected")
                return None
            
            # Check for speech ended event
            if "type" in response and response["type"] == "UtteranceEnd":
                logger.debug("Utterance end detected")
                return None
            
            # Process transcription results
            if "channel" in response and "alternatives" in response["channel"]:
                alternatives = response["channel"]["alternatives"]
                if alternatives:
                    # Get the first (best) alternative
                    alternative = alternatives[0]
                    transcript = alternative.get("transcript", "")
                    confidence = alternative.get("confidence", 0.0)
                    is_final = response.get("is_final", False)
                    words = alternative.get("words", [])
                    
                    # Create result object
                    result = StreamingTranscriptionResult(
                        text=transcript,
                        is_final=is_final,
                        confidence=confidence,
                        words=words,
                        chunk_id=self.utterance_id
                    )
                    
                    if is_final:
                        self.utterance_id += 1
                        self.last_result = result
                    
                    # Call callback if provided
                    if callback and (transcript.strip() or is_final):
                        await callback(result)
                    
                    return result if is_final else None
            
            return None
            
        except asyncio.TimeoutError:
            # No messages available, which is normal
            return None
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error while processing messages: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")
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
                    if result and result.is_final:
                        final_results.append(result)
                    
                    # Simulate real-time streaming if requested
                    if simulate_realtime:
                        await asyncio.sleep(chunk_size / self.sample_rate / 2)  # Half real-time speed
            
            # Close the stream
            await self.stop_streaming()
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error streaming audio file: {str(e)}")
            await self.stop_streaming()
            raise STTStreamingError(f"Error streaming audio file: {str(e)}")