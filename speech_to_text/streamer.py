"""
Streaming functionality for Google Cloud Speech-to-Text.

This module provides streaming transcription capabilities using Google Cloud Speech-to-Text,
replacing the Deepgram dependencies.
"""
import logging
import time
import asyncio
from typing import Dict, Any, Optional, AsyncIterator, List, Callable, Awaitable, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0

class GoogleCloudStreamer:
    """
    Streaming wrapper for Google Cloud Speech-to-Text.
    
    This class handles real-time streaming of audio to Google Cloud Speech-to-Text,
    providing a convenient interface for processing audio chunks and receiving transcriptions.
    """
    
    def __init__(
        self,
        speech_client,
        language_code: str = "en-US",
        model: str = "phone_call",
        use_enhanced: bool = True,
        sample_rate: int = 16000,
        enable_automatic_punctuation: bool = True
    ):
        """
        Initialize the streamer.
        
        Args:
            speech_client: Google Cloud Speech-to-Text client
            language_code: Language code for recognition
            model: Recognition model
            use_enhanced: Whether to use enhanced model
            sample_rate: Audio sample rate in Hz
            enable_automatic_punctuation: Whether to enable automatic punctuation
        """
        self.speech_client = speech_client
        self.language_code = language_code
        self.model = model
        self.use_enhanced = use_enhanced
        self.sample_rate = sample_rate
        self.enable_automatic_punctuation = enable_automatic_punctuation
        
        # Streaming state
        self.is_streaming = False
        self.audio_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.stream_task = None
        self.last_result = None
        self.chunk_id = 0
        
        logger.info(f"Initialized Google Cloud Streamer with model={model}, language={language_code}")
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        if self.is_streaming:
            logger.info("Streaming session already active, skipping")
            return
            
        # Reset state
        self.is_streaming = True
        self.audio_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.last_result = None
        self.chunk_id = 0
        
        # Start streaming task
        self.stream_task = asyncio.create_task(self._streaming_loop())
        logger.info("Started new streaming session")
    
    async def _streaming_loop(self) -> None:
        """Process streaming audio and handle transcription results."""
        try:
            # Create audio stream generator
            async def audio_generator():
                while self.is_streaming:
                    try:
                        # Get audio chunk with timeout
                        audio_chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=10.0)
                        
                        # Check for end signal
                        if audio_chunk is None:
                            break
                            
                        yield audio_chunk
                        self.audio_queue.task_done()
                    except asyncio.TimeoutError:
                        # No new audio received, check if we should continue
                        if not self.is_streaming:
                            break
                        # Yield empty chunk to keep stream alive
                        yield b''
                    except Exception as e:
                        logger.error(f"Error in audio generator: {e}")
                        break
            
            # Start streaming recognition
            streaming_results = await self.speech_client.streaming_recognize(audio_generator())
            
            # Process results
            async for result in streaming_results:
                if not self.is_streaming:
                    break
                    
                # Update last result
                if result:
                    # Convert to StreamingTranscriptionResult
                    self.chunk_id += 1
                    transcription_result = StreamingTranscriptionResult(
                        text=result.get("transcription", ""),
                        is_final=result.get("is_final", False),
                        confidence=result.get("confidence", 0.0),
                        start_time=time.time() - 5.0,  # Approximate start time
                        end_time=time.time(),
                        chunk_id=self.chunk_id
                    )
                    
                    # Update last result
                    self.last_result = transcription_result
                    
                    # Put in results queue
                    await self.results_queue.put(transcription_result)
                
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            self.is_streaming = False
    
    async def process_audio_chunk(
        self,
        audio_chunk: bytes,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process an audio chunk through the streaming recognition.
        
        Args:
            audio_chunk: Audio data chunk
            callback: Optional callback for results
            
        Returns:
            The latest transcription result, if available
        """
        if not self.is_streaming:
            logger.warning("Streaming session not active, starting new session")
            await self.start_streaming()
        
        # Add to queue
        await self.audio_queue.put(audio_chunk)
        
        # Check for results
        result = None
        while not self.results_queue.empty():
            result = await self.results_queue.get()
            # Call callback if provided
            if callback and result:
                await callback(result)
            self.results_queue.task_done()
        
        return result or self.last_result
    
    async def stop_streaming(self) -> Optional[StreamingTranscriptionResult]:
        """
        Stop the streaming session.
        
        Returns:
            The final transcription result, if available
        """
        if not self.is_streaming:
            return None
            
        # Set flag to stop streaming
        self.is_streaming = False
        
        # Signal end of audio
        await self.audio_queue.put(None)
        
        # Wait for stream task to complete
        if self.stream_task:
            try:
                # Wait with timeout to avoid hanging
                await asyncio.wait_for(asyncio.shield(self.stream_task), timeout=5.0)
            except asyncio.TimeoutError:
                # Cancel task if timeout
                if not self.stream_task.done():
                    self.stream_task.cancel()
                    try:
                        await self.stream_task
                    except asyncio.CancelledError:
                        pass
            except Exception as e:
                logger.error(f"Error stopping stream task: {e}")
            
            self.stream_task = None
        
        logger.info("Stopped streaming session")
        return self.last_result