"""
Streaming implementation for Google Cloud Speech-to-Text.
"""
import asyncio
import logging
import queue
import threading
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Awaitable

from .client import GoogleCloudSTT
from .config import config
from .exceptions import STTStreamingError

logger = logging.getLogger(__name__)

class STTStreamer:
    """
    Manages real-time streaming of audio to Google Cloud STT.
    
    Optimized for low-latency telephony applications, this class handles
    streaming audio to the STT API and processing results in real-time.
    """
    
    def __init__(
        self,
        stt_client: Optional[GoogleCloudSTT] = None,
        **stt_kwargs
    ):
        """
        Initialize the STT streamer.
        
        Args:
            stt_client: Existing GoogleCloudSTT client, or one will be created
            **stt_kwargs: Arguments to pass to GoogleCloudSTT if creating a new client
        """
        self.stt_client = stt_client or GoogleCloudSTT(**stt_kwargs)
        self.audio_queue = asyncio.Queue()
        self.running = False
        self.stream_task = None
        
        # Buffer for audio chunks
        self.buffer = bytearray()
        self.buffer_lock = asyncio.Lock()
        self.min_buffer_size = 1024  # Minimum size to process
        
        # Barge-in detection
        self.speech_detected_event = asyncio.Event()
        self.speech_detected_callback = None
        
        # Create event to track when streaming is fully started
        self.streaming_started = asyncio.Event()
        self.streaming_started.clear()
    
    async def _audio_generator(self) -> AsyncGenerator[bytes, None]:
        """
        Generate audio chunks from the queue for streaming to Google STT.
        
        Yields:
            Audio chunks as they become available
        """
        yield b''  # Empty chunk to start the stream
        
        while self.running:
            try:
                # Get audio from queue with timeout
                audio_chunk = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=5.0  # 5 second timeout
                )
                
                # Check for end signal
                if audio_chunk is None:
                    break
                
                yield audio_chunk
                self.audio_queue.task_done()
            except asyncio.TimeoutError:
                # No new audio received within timeout
                if self.running:
                    # Send an empty chunk to keep the stream alive
                    yield b''
                else:
                    break
            except Exception as e:
                logger.error(f"Error in audio generator: {str(e)}")
                if self.running:
                    raise STTStreamingError(f"Audio streaming error: {str(e)}")
    
    async def process_audio_chunk(
        self, 
        audio_chunk: bytes,
        callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process an audio chunk through the streaming recognition.
        
        Args:
            audio_chunk: Audio data chunk
            callback: Optional callback for results
            
        Returns:
            The latest transcription result, if available
        """
        if not self.running:
            raise STTStreamingError("Cannot process audio: streamer is not running")
        
        async with self.buffer_lock:
            # Add to buffer
            self.buffer.extend(audio_chunk)
            
            # Process buffer if it's large enough
            if len(self.buffer) >= self.min_buffer_size:
                # Copy and clear buffer
                chunk_to_process = bytes(self.buffer)
                self.buffer = bytearray()
                
                # Add to queue
                await self.audio_queue.put(chunk_to_process)
                
                # Return latest result if available
                if hasattr(self, 'latest_result'):
                    return self.latest_result
        
        return None
    
    async def start_streaming(
        self,
        speech_detected_callback: Optional[Callable[[bool], Awaitable[None]]] = None
    ) -> None:
        """
        Start the STT streaming process.
        
        Args:
            speech_detected_callback: Optional callback when speech is detected
        """
        if self.running:
            return
            
        self.running = True
        self.speech_detected_event.clear()
        self.speech_detected_callback = speech_detected_callback
        self.buffer = bytearray()
        self.latest_result = None
        
        # Create audio generator
        audio_stream = self._audio_generator()
        
        # Start streaming in a separate task
        self.stream_task = asyncio.create_task(self._stream_processor(audio_stream))
        
        # Wait for streaming to fully start
        self.streaming_started.set()
        
        logger.info("Started Google Cloud STT streaming session")
    
    async def _stream_processor(self, audio_stream: AsyncGenerator[bytes, None]) -> None:
        """
        Process the streaming recognition and handle results.
        
        Args:
            audio_stream: Audio stream generator
        """
        try:
            # Process streaming recognition
            async for result in self.stt_client.streaming_recognize(audio_stream):
                # Store latest result
                self.latest_result = result
                
                # Check for speech detection
                if result.get("is_final", False) or (
                    result.get("stability", 0) > 0.8 and 
                    result.get("transcription", "") and 
                    result.get("confidence", 0) > 0.7
                ):
                    # Set speech detected event
                    self.speech_detected_event.set()
                    
                    # Call speech detected callback if provided
                    if self.speech_detected_callback:
                        await self.speech_detected_callback(True)
                
                # Log result
                if result.get("is_final", False):
                    logger.info(f"Final transcription: {result.get('transcription', '')}")
                elif self.speech_detected_event.is_set():
                    logger.debug(f"Interim result: {result.get('transcription', '')}")
                
        except Exception as e:
            logger.error(f"Error in stream processor: {e}")
        finally:
            self.running = False
            self.streaming_started.clear()
    
    async def stop_streaming(self) -> Optional[Dict[str, Any]]:
        """
        Stop the STT streaming process.
        
        Returns:
            The final transcription result, if available
        """
        if not self.running:
            return None
            
        self.running = False
        
        # Signal end of audio
        await self.audio_queue.put(None)
        
        # Wait for stream task to complete
        if self.stream_task:
            try:
                await asyncio.wait_for(self.stream_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for stream task to complete")
            except Exception as e:
                logger.error(f"Error waiting for stream task: {e}")
            
            self.stream_task = None
        
        # Clear events
        self.speech_detected_event.clear()
        self.streaming_started.clear()
        
        # Return final result
        return self.latest_result

class SpeechDetector:
    """
    Enhanced speech and barge-in detection using Google Cloud STT results.
    """
    
    def __init__(
        self,
        energy_threshold: float = 0.03,
        min_speech_duration_ms: int = 300,
        cooldown_ms: int = 1500
    ):
        """
        Initialize speech detector.
        
        Args:
            energy_threshold: Energy threshold for speech detection
            min_speech_duration_ms: Minimum duration to confirm speech
            cooldown_ms: Cooldown period after agent speech
        """
        self.energy_threshold = energy_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.cooldown_ms = cooldown_ms
        
        # State tracking
        self.agent_speaking = False
        self.agent_speech_start_time = 0
        self.user_speech_start_time = 0
        self.is_speech_detected = False
        self.last_energy = 0
        
        # Barge-in tracking
        self.barge_in_detected = False
        self.last_barge_in_time = 0
    
    def check_energy_threshold(self, audio_data: bytes) -> float:
        """
        Calculate audio energy and check if it exceeds the threshold.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Audio energy level
        """
        # Convert bytes to samples
        import struct
        import numpy as np
        
        # Assuming 16-bit PCM audio
        format_str = f"{len(audio_data)//2}h"
        try:
            samples = struct.unpack(format_str, audio_data)
            
            # Convert to numpy array and normalize
            audio_array = np.array(samples, dtype=np.float32) / 32768.0
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(np.square(audio_array)))
            
            # Store last energy
            self.last_energy = energy
            
            return energy
        except Exception as e:
            logger.error(f"Error calculating audio energy: {e}")
            return 0
    
    def detect_speech(
        self, 
        audio_data: bytes,
        transcription_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Detect speech using energy levels and transcription results.
        
        Args:
            audio_data: Audio data as bytes
            transcription_result: Optional transcription result
            
        Returns:
            True if speech detected
        """
        import time
        current_time = time.time() * 1000  # Convert to ms
        
        # Check energy level
        energy = self.check_energy_threshold(audio_data)
        energy_based_detection = energy > self.energy_threshold
        
        # Check transcription result
        transcription_based_detection = False
        if transcription_result:
            # Check if we have a good transcription
            has_text = transcription_result.get("transcription", "").strip() != ""
            good_confidence = transcription_result.get("confidence", 0) > 0.6
            is_final = transcription_result.get("is_final", False)
            
            transcription_based_detection = (
                has_text and (good_confidence or is_final)
            )
        
        # Combine detection methods
        is_speech = energy_based_detection or transcription_based_detection
        
        # State tracking
        if is_speech and not self.is_speech_detected:
            # Speech start
            self.is_speech_detected = True
            self.user_speech_start_time = current_time
        elif not is_speech and self.is_speech_detected:
            # Check if speech duration was long enough
            speech_duration = current_time - self.user_speech_start_time
            if speech_duration < self.min_speech_duration_ms:
                # Too short, likely noise
                self.is_speech_detected = False
        
        return self.is_speech_detected
    
    def check_for_barge_in(
        self,
        audio_data: bytes,
        transcription_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user is barging in while agent is speaking.
        
        Args:
            audio_data: Audio data as bytes
            transcription_result: Optional transcription result
            
        Returns:
            True if barge-in detected
        """
        if not self.agent_speaking:
            return False
            
        import time
        current_time = time.time() * 1000  # Convert to ms
        
        # Check cooldown period
        time_since_agent_started = current_time - self.agent_speech_start_time
        if time_since_agent_started < self.cooldown_ms:
            return False
            
        # Check if enough time has passed since last barge-in
        time_since_last_barge_in = current_time - self.last_barge_in_time
        if time_since_last_barge_in < 800:  # 800ms minimum between barge-ins
            return False
        
        # Detect speech with higher threshold for barge-in
        energy = self.check_energy_threshold(audio_data)
        energy_based_detection = energy > (self.energy_threshold * 1.5)
        
        # Check transcription result for strong confidence
        transcription_based_detection = False
        if transcription_result:
            transcription = transcription_result.get("transcription", "")
            confidence = transcription_result.get("confidence", 0)
            
            transcription_based_detection = (
                transcription.strip() != "" and confidence > 0.7
            )
        
        # Combine detection methods with stricter criteria for barge-in
        is_barge_in = energy_based_detection or transcription_based_detection
        
        if is_barge_in:
            # Update barge-in state
            self.barge_in_detected = True
            self.last_barge_in_time = current_time
            logger.info("Barge-in detected!")
        
        return is_barge_in
    
    def set_agent_speaking(self, is_speaking: bool) -> None:
        """
        Set the agent speaking state.
        
        Args:
            is_speaking: Whether the agent is speaking
        """
        import time
        
        # Only update if state changes
        if is_speaking != self.agent_speaking:
            self.agent_speaking = is_speaking
            
            if is_speaking:
                # Agent started speaking
                self.agent_speech_start_time = time.time() * 1000  # Convert to ms
                self.barge_in_detected = False
                logger.info("Agent speaking started")
            else:
                # Agent stopped speaking
                logger.info("Agent speaking stopped")