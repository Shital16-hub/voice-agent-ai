"""
Google Cloud Speech-to-Text client for streaming transcription.
"""
import datetime
import os
import logging
import asyncio
import time
import queue
import threading
from typing import Dict, Any, Optional, Union, List, AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass

from google.cloud import speech_v1 as speech
from google.oauth2 import service_account

from ..config import config
from .models import TranscriptionConfig
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

class GoogleCloudStreamingSTT:
    """
    Client for the Google Cloud Speech-to-Text streaming API, optimized for telephony.
    
    This class handles real-time streaming STT using Google Cloud's API,
    with optimizations for telephony applications.
    """
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None,
        language_code: Optional[str] = None,
        sample_rate: Optional[int] = None,
        encoding: str = "LINEAR16",
        channels: int = 1,
        interim_results: Optional[bool] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the Google Cloud STT streaming client.
        
        Args:
            credentials_path: Path to Google credentials JSON
            language_code: Language code for recognition
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            channels: Number of audio channels
            interim_results: Whether to return interim results
            model: Model to use (defaults to phone_call)
        """
        # Use same credentials path as TTS if not provided
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not self.credentials_path:
            raise ValueError("Google Cloud credentials path is required. Set GOOGLE_APPLICATION_CREDENTIALS in .env or pass directly.")
        
        self.language_code = language_code or config.language
        self.sample_rate = sample_rate or 8000  # Default to 8kHz for telephony
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results if interim_results is not None else True
        self.model = model or "phone_call"
        
        # Streaming state
        self.is_streaming = False
        self.stream = None
        self.streaming_config = None
        self.streaming_lock = asyncio.Lock()
        self.streaming_started = asyncio.Event()
        self.streaming_started.clear()
        
        # Threading resources for streaming
        self.audio_queue = None
        self.result_queue = None
        self.streaming_thread = None
        self.stop_event = None
        
        # Result tracking
        self.utterance_id = 0
        self.last_result = None
        
        # Initialize client
        self._init_client()
        
        logger.info(f"Initialized Google Cloud STT streaming client with model={self.model}, "
                  f"language={self.language_code}, sample_rate={self.sample_rate}")
    
    def _init_client(self):
        """Initialize the Google Cloud STT client with the provided credentials."""
        try:
            # Check if credentials file exists
            if not os.path.exists(self.credentials_path):
                raise ValueError(f"Google Cloud credentials file not found: {self.credentials_path}")
                
            # Initialize credentials
            credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
            
            # Create the client
            self.client = speech.SpeechClient(credentials=credentials)
            logger.info(f"Initialized Google Cloud STT client with language: {self.language_code}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud STT client: {e}")
            raise STTAPIError(f"Failed to initialize Google Cloud STT client: {str(e)}")
    
    def _get_streaming_config(self, config_obj: Optional[TranscriptionConfig] = None) -> Dict[str, Any]:
        """Get streaming recognition configuration."""
        encoding_map = {
            "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "FLAC": speech.RecognitionConfig.AudioEncoding.FLAC,
            "MP3": speech.RecognitionConfig.AudioEncoding.MP3,
            "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
        }
        
        # Start with default settings optimized for telephony
        recognition_config = {
            "language_code": self.language_code,
            "sample_rate_hertz": self.sample_rate,
            "encoding": encoding_map.get(self.encoding, encoding_map["LINEAR16"]),
            "audio_channel_count": self.channels,
            "enable_automatic_punctuation": True,
            "enable_word_time_offsets": True,
            "model": self.model,
            "use_enhanced": True,
            "max_alternatives": 1
        }
        
        # Add speech contexts for improved recognition of keywords
        speech_contexts = [{
            "phrases": ["price", "plan", "cost", "subscription", "service", "features", "support",
                       "help", "agent", "assistant", "voice", "stop", "continue"],
            "boost": 15.0
        }]
        recognition_config["speech_contexts"] = speech_contexts
        
        # Create streaming config
        streaming_config = {
            "config": recognition_config,
            "interim_results": self.interim_results
        }
        
        # Override with config object if provided
        if config_obj:
            config_dict = config_obj.dict(exclude_none=True, exclude_unset=True)
            if "encoding" in config_dict and config_dict["encoding"] in encoding_map:
                recognition_config["encoding"] = encoding_map[config_dict["encoding"]]
                
            for key, value in config_dict.items():
                if key != "encoding":
                    recognition_config[key] = value
        
        return streaming_config
    
    async def start_streaming(self, config_obj: Optional[TranscriptionConfig] = None) -> None:
        """
        Start a streaming session.
        
        Args:
            config_obj: Optional configuration object
        """
        async with self.streaming_lock:
            if self.is_streaming:
                await self.stop_streaming()
            
            # Set up streaming configuration
            self.streaming_config = self._get_streaming_config(config_obj)
            
            # Set up threading resources
            self.audio_queue = queue.Queue()
            self.result_queue = queue.Queue()
            self.stop_event = threading.Event()
            
            # Start streaming thread
            self.streaming_thread = threading.Thread(
                target=self._run_streaming_thread,
                daemon=True
            )
            self.streaming_thread.start()
            
            # Wait for streaming to be ready
            self.is_streaming = True
            self.streaming_started.set()
            
            logger.info("Started Google Cloud STT streaming session")
    
    def _run_streaming_thread(self):
        """Run the streaming recognition in a background thread."""
        try:
            # Create a new speech client for this thread
            credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
            client = speech.SpeechClient(credentials=credentials)
            
            # For debugging - print the client version and methods
            logger.info(f"Using Google Cloud Speech version: {speech.__version__}")
            
            # Create the recognition config
            rec_config = speech.RecognitionConfig(**self.streaming_config["config"])
            
            # Create the streaming config - this is what we'll pass as the config parameter
            streaming_config = speech.StreamingRecognitionConfig(
                config=rec_config,
                interim_results=self.streaming_config["interim_results"]
            )
            
            # Create an audio generator that yields requests with ONLY audio content
            # The config will be passed separately
            def request_generator():
                # Process audio chunks from queue - no config in these requests
                while not self.stop_event.is_set():
                    # Get audio chunk from queue with timeout
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.5)
                        if audio_chunk:
                            # All requests contain only audio
                            yield speech.StreamingRecognizeRequest(audio_content=audio_chunk)
                        self.audio_queue.task_done()
                    except queue.Empty:
                        # No audio available yet
                        continue
                    except Exception as e:
                        logger.error(f"Error processing audio chunk: {e}")
                        break
            
            # Call streaming_recognize with BOTH required parameters:
            # 1. config - The streaming config
            # 2. requests - The generator that produces only audio content requests
            response_stream = client.streaming_recognize(
                config=streaming_config,
                requests=request_generator()
            )
            
            # Process responses
            for response in response_stream:
                if self.stop_event.is_set():
                    break
                    
                # Skip empty responses
                if not response.results:
                    continue
                    
                # Process each result
                for result in response.results:
                    if not result.alternatives:
                        continue
                        
                    # Create a transcription result
                    self.utterance_id += 1
                    
                    # Extract word timings if available
                    words = []
                    
                    # Capture end time if available
                    if hasattr(result, 'result_end_time') and result.result_end_time:
                        # Fix for handling timedelta objects
                        if isinstance(result.result_end_time, datetime.timedelta):
                            # For timedelta objects, use total_seconds()
                            end_time = result.result_end_time.total_seconds()
                        else:
                            # For the old format with seconds and nanos
                            try:
                                end_time = result.result_end_time.seconds + result.result_end_time.nanos / 1e9
                            except AttributeError:
                                # Fallback if neither format works
                                end_time = time.time()
                    else:
                        end_time = time.time()
                    
                    # Create the result object
                    transcription_result = StreamingTranscriptionResult(
                        text=result.alternatives[0].transcript,
                        is_final=result.is_final,
                        confidence=result.alternatives[0].confidence if result.is_final else 0.0,
                        start_time=0.0,  # Start time not available from Google Cloud
                        end_time=end_time,
                        chunk_id=self.utterance_id,
                        words=words
                    )
                    
                    # Add to result queue for async processing
                    self.result_queue.put(transcription_result)
                    
                    # Store as last result if final
                    if result.is_final:
                        self.last_result = transcription_result
                    
        except Exception as e:
            logger.error(f"Error in streaming thread: {e}", exc_info=True)
            # Add error to result queue
            error_result = StreamingTranscriptionResult(
                text="",
                is_final=True,
                confidence=0.0,
                start_time=0.0,
                end_time=time.time(),
                chunk_id=self.utterance_id + 1,
                words=[]
            )
            self.result_queue.put(error_result)
    
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
        # Check if streaming is initialized
        if not self.is_streaming:
            # Auto-initialize streaming if needed
            try:
                await self.start_streaming()
                # Wait for streaming to be properly initialized
                await asyncio.wait_for(self.streaming_started.wait(), timeout=5.0)
            except Exception as e:
                raise STTStreamingError(f"Failed to auto-initialize streaming: {e}")
        
        try:
            # Ensure audio is in bytes format
            if not isinstance(audio_chunk, (bytes, bytearray, memoryview)):
                raise ValueError("Audio chunk must be bytes, bytearray, or memoryview")
            
            # Add to audio queue
            self.audio_queue.put(bytes(audio_chunk))
            
            # Check for results
            results = []
            
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    results.append(result)
                    self.result_queue.task_done()
                    
                    # Call callback if provided
                    if callback:
                        await callback(result)
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    break
                
            # Return the most recent final result or any interim result
            if results:
                # First try to find final results
                final_results = [r for r in results if r.is_final]
                if final_results:
                    return final_results[-1]  # Return the most recent final result
                
                # If no final results, return the last interim result
                return results[-1]
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            raise STTStreamingError(f"Error processing audio chunk: {e}")
    
    async def stop_streaming(self) -> Optional[tuple[str, float]]:
        """
        Stop the streaming session.
        
        Returns:
            Optional tuple of (transcription, duration) or None on error
        """
        if not self.is_streaming:
            logger.warning("stop_streaming called but not currently streaming")
            return None
        
        try:
            # Signal thread to stop
            if self.stop_event:
                self.stop_event.set()
            
            # Wait for thread to finish
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=2.0)
            
            # Clean up
            self.is_streaming = False
            self.streaming_started.clear()
            
            # Return transcription if available
            if self.last_result:
                return self.last_result.text, self.last_result.end_time
            
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return None
        finally:
            # Reset threading resources
            self.audio_queue = None
            self.result_queue = None
            self.streaming_thread = None
            self.stop_event = None
    
    async def stream_audio_file(
        self,
        file_path: str,
        chunk_size: int = 4096,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None,
        simulate_realtime: bool = False
    ) -> List[StreamingTranscriptionResult]:
        """
        Stream an audio file to Google Cloud STT.
        
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
            
            # Read the audio file
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
            logger.error(f"Error streaming audio file: {e}")
            await self.stop_streaming()
            raise STTStreamingError(f"Error streaming audio file: {str(e)}")