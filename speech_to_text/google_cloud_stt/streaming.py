# speech_to_text/google_cloud_stt/streaming.py
"""
Google Cloud Speech-to-Text client for streaming recognition.

Optimized for real-time telephony applications with barge-in detection.
"""
import os
import logging
import asyncio
import time
import re
from typing import Optional, Dict, Any, Callable, Awaitable, List, AsyncIterator
import numpy as np
from dataclasses import dataclass
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Google Cloud Speech imports
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

from ..config import config
from .exceptions import STTError, STTStreamingError

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

class GoogleCloudStreamingSTT:
    """
    Client for Google Cloud Speech-to-Text streaming API, optimized for telephony.
    
    This class handles real-time streaming STT operations using Google Cloud Speech API,
    with configurations optimized for telephony applications and barge-in detection.
    """
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        interim_results: bool = True,
        model: str = "phone_call",
        enhanced: bool = True
    ):
        """
        Initialize the Google Cloud Streaming STT client.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            language: Language code for recognition (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            interim_results: Whether to return interim results
            model: Speech model to use (phone_call, default, command_and_search, etc.)
            enhanced: Whether to use enhanced models
        """
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.interim_results = interim_results
        self.model = model
        self.enhanced = enhanced
        
        # Set up credentials
        self._setup_credentials()
        
        # State management
        self.is_streaming = False
        self.streaming_client = None
        self.streaming_config = None
        self.streaming_responses = None
        self.streaming_thread = None
        self.stream_executor = None
        self.audio_queue = None
        self.response_queue = None
        self.stop_event = None
        self.chunk_id_counter = 0
        self.last_result = None
        
        # Barge-in detection
        self.barge_in_detected = False
        self.last_activity_time = 0
        self.consecutive_speech_chunks = 0
        self.min_barge_in_energy = 0.1
        self.min_consecutive_chunks = 3
        
        logger.info(f"Initialized Google Cloud Streaming STT with language: {self.language}")
    
    def _setup_credentials(self):
        """Set up Google Cloud credentials."""
        if self.credentials_path:
            try:
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                logger.info(f"Using credentials from {self.credentials_path}")
            except Exception as e:
                logger.error(f"Error loading credentials: {e}")
                self.credentials = None
        else:
            logger.info("Using default application credentials")
            self.credentials = None
        
        # Initialize client
        self.client = speech.SpeechClient(credentials=self.credentials)
    
    def _get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """
        Get streaming recognition configuration.
        
        Returns:
            StreamingRecognitionConfig for Google Cloud Speech API
        """
        # Create recognition config with telephony optimizations
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            model=self.model,
            use_enhanced=self.enhanced,
            audio_channel_count=1,
            enable_separate_recognition_per_channel=False,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            # Telephony optimizations
            speech_contexts=[speech.SpeechContext(
                phrases=config.keywords,
                boost=10.0
            )],
        )
        
        # Create streaming config
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self.interim_results,
            # These settings optimize for real-time telephony
            single_utterance=False,  # Process multiple utterances
        )
        
        return streaming_config
    
    def _create_streaming_request_generator(self):
        """
        Create generator for streaming requests.
        
        Returns:
            Generator for sending streaming requests
        """
        # First request with config
        yield speech.StreamingRecognizeRequest(
            streaming_config=self.streaming_config
        )
        
        # Process audio chunks from queue
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with timeout
                chunk = self.audio_queue.get(timeout=0.5)
                
                # Check for stop signal
                if chunk is None:
                    break
                
                # Create request with audio content
                request = speech.StreamingRecognizeRequest(audio_content=chunk)
                yield request
                
                self.audio_queue.task_done()
            except Exception as e:
                if "queue.Empty" in str(e.__class__):
                    # No data available, continue
                    continue
                logger.error(f"Error in streaming request generator: {e}")
                break
    
    def _process_streaming_responses(self):
        """Process streaming responses in a separate thread."""
        try:
            for response in self.streaming_responses:
                # Check if stopped
                if self.stop_event.is_set():
                    break
                
                # Skip empty response
                if not response.results:
                    continue
                
                # Process each result
                for result in response.results:
                    # Skip empty alternatives
                    if not result.alternatives:
                        continue
                    
                    # Get the first alternative
                    alternative = result.alternatives[0]
                    
                    # Create result object
                    self.chunk_id_counter += 1
                    transcript = alternative.transcript
                    confidence = alternative.confidence if result.is_final else 0.5
                    
                    # Extract word timing if available
                    words = []
                    if hasattr(alternative, 'words') and alternative.words:
                        for word_info in alternative.words:
                            words.append({
                                "word": word_info.word,
                                "start_time": word_info.start_time.total_seconds(),
                                "end_time": word_info.end_time.total_seconds(),
                                "confidence": confidence  # Per-word confidence not provided
                            })
                    
                    # Calculate timing
                    start_time = time.time() - 0.5  # Estimated
                    end_time = time.time()
                    
                    # Create result object
                    transcription_result = StreamingTranscriptionResult(
                        text=transcript,
                        is_final=result.is_final,
                        confidence=confidence,
                        start_time=start_time,
                        end_time=end_time,
                        chunk_id=self.chunk_id_counter,
                        words=words
                    )
                    
                    # Update last result
                    self.last_result = transcription_result
                    
                    # Check for barge-in based on energy
                    if transcript and (confidence > 0.6 or result.is_final):
                        self.consecutive_speech_chunks += 1
                        if self.consecutive_speech_chunks >= self.min_consecutive_chunks:
                            self.barge_in_detected = True
                            logger.info(f"Barge-in detected: {transcript}")
                    else:
                        self.consecutive_speech_chunks = 0
                    
                    # Update last activity time
                    self.last_activity_time = time.time()
                    
                    # Add to response queue
                    self.response_queue.put(transcription_result)
        except Exception as e:
            logger.error(f"Error processing streaming responses: {e}")
            # Signal error
            self.response_queue.put(None)
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        # Check if already streaming
        if self.is_streaming:
            logger.warning("Already streaming, stopping previous session")
            await self.stop_streaming()
        
        logger.info("Starting Google Cloud Speech streaming session")
        
        # Reset state
        self.is_streaming = True
        self.barge_in_detected = False
        self.consecutive_speech_chunks = 0
        self.last_activity_time = time.time()
        self.chunk_id_counter = 0
        self.last_result = None
        
        # Create thread-safe queues
        self.audio_queue = Queue()
        self.response_queue = Queue()
        self.stop_event = threading.Event()
        
        # Create thread pool executor
        self.stream_executor = ThreadPoolExecutor(max_workers=2)
        
        # Get streaming config
        self.streaming_config = self._get_streaming_config()
        
        # Start streaming in a separate thread
        self.streaming_thread = self.stream_executor.submit(self._run_streaming)
    
    def _run_streaming(self):
        """Run streaming recognition in a separate thread."""
        try:
            # Create request generator
            request_generator = self._create_streaming_request_generator()
            
            # Start streaming recognition
            self.streaming_responses = self.client.streaming_recognize(
                requests=request_generator,
                timeout=60  # 1 minute timeout
            )
            
            # Process responses
            self._process_streaming_responses()
            
        except Exception as e:
            logger.error(f"Error in streaming thread: {e}")
            # Signal error
            self.response_queue.put(None)
    
    async def process_audio_chunk(
        self, 
        audio_chunk: bytes,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process an audio chunk.
        
        Args:
            audio_chunk: Audio data as bytes
            callback: Optional async callback for results
            
        Returns:
            Optional transcription result
        """
        if not self.is_streaming:
            logger.warning("Not streaming, call start_streaming() first")
            return None
        
        try:
            # Check for barge-in detection
            if len(audio_chunk) > 0:
                # Convert to numpy array for energy detection
                if isinstance(audio_chunk, bytes):
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_array = audio_chunk
                
                # Calculate energy
                energy = np.mean(np.abs(audio_array))
                
                # Check for potential barge-in
                if energy > self.min_barge_in_energy:
                    logger.debug(f"Potential barge-in detected, energy: {energy:.3f}")
                    self.consecutive_speech_chunks += 1
                    if self.consecutive_speech_chunks >= self.min_consecutive_chunks:
                        self.barge_in_detected = True
                        logger.info("Barge-in confirmed, high energy for multiple chunks")
                else:
                    self.consecutive_speech_chunks = 0
            
            # Add to audio queue
            self.audio_queue.put(audio_chunk)
            
            # Check response queue for results
            results = []
            while not self.response_queue.empty():
                result = self.response_queue.get()
                if result is None:
                    # Error occurred
                    logger.error("Error in streaming recognition")
                    return None
                
                results.append(result)
                self.response_queue.task_done()
            
            # Process results
            latest_result = None
            for result in results:
                if callback:
                    await callback(result)
                
                # Keep track of latest final result
                if result.is_final:
                    latest_result = result
            
            return latest_result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def stop_streaming(self) -> tuple[str, float]:
        """
        Stop the streaming session and return final results.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.is_streaming:
            logger.warning("Not streaming, nothing to stop")
            return "", 0.0
        
        logger.info("Stopping Google Cloud Speech streaming session")
        
        # Signal stop
        self.stop_event.set()
        self.audio_queue.put(None)
        
        # Wait for remaining results with timeout
        try:
            final_results = []
            
            # Try to get any final results with timeout
            start_wait = time.time()
            max_wait = 2.0  # Max 2 seconds to wait for final results
            
            while time.time() - start_wait < max_wait:
                try:
                    # Get with short timeout
                    result = self.response_queue.get(timeout=0.1)
                    if result is None:
                        break
                    
                    final_results.append(result)
                    self.response_queue.task_done()
                except Exception:
                    # Timeout, check if queue is likely to be empty
                    if self.response_queue.empty():
                        # Wait a bit more then break
                        await asyncio.sleep(0.2)
                        break
            
            # Get the best final result
            final_text = ""
            duration = 0.0
            
            if final_results:
                # Use the latest final result
                final_results = [r for r in final_results if r.is_final]
                if final_results:
                    best_result = max(final_results, key=lambda r: r.confidence)
                    final_text = best_result.text
                    duration = best_result.end_time - best_result.start_time
            
            # Use last result if no final results
            if not final_text and self.last_result:
                final_text = self.last_result.text
                duration = self.last_result.end_time - self.last_result.start_time
            
        finally:
            # Clean up resources
            if self.streaming_thread:
                try:
                    # Wait with timeout
                    self.streaming_thread.result(timeout=1.0)
                except Exception:
                    pass
            
            if self.stream_executor:
                self.stream_executor.shutdown(wait=False)
            
            # Reset state
            self.is_streaming = False
            self.streaming_responses = None
            self.streaming_thread = None
            self.stream_executor = None
            self.audio_queue = None
            self.response_queue = None
            self.stop_event = None
        
        return final_text, duration
    
    def is_barge_in_detected(self) -> bool:
        """
        Check if barge-in has been detected.
        
        Returns:
            True if barge-in has been detected
        """
        return self.barge_in_detected
    
    def reset_barge_in_detection(self) -> None:
        """Reset barge-in detection state."""
        self.barge_in_detected = False
        self.consecutive_speech_chunks = 0