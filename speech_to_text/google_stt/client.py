"""
Google Cloud Speech-to-Text client for Voice AI Agent.
"""
import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
import io
import wave

from google.cloud import speech
from google.oauth2 import service_account

from .config import config
from .exceptions import STTError, STTAPIError

logger = logging.getLogger(__name__)

class GoogleCloudSTT:
    """
    Client for Google Cloud Speech-to-Text API with streaming capabilities,
    optimized for telephony applications.
    
    Provides real-time transcription with enhanced phone call model
    and noise cancellation.
    """
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        language_code: Optional[str] = None,
        sample_rate: Optional[int] = None,
        enable_automatic_punctuation: Optional[bool] = None,
        use_enhanced: Optional[bool] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Google Cloud STT client.
        
        Args:
            credentials_path: Path to Google credentials JSON
            language_code: Language code for recognition
            sample_rate: Audio sample rate in Hz
            enable_automatic_punctuation: Whether to add punctuation
            use_enhanced: Whether to use enhanced model
            model: Model to use (e.g., phone_call, video, default)
        """
        self.credentials_path = credentials_path or config.google_credentials_path
        self.language_code = language_code or config.language_code
        self.sample_rate = sample_rate or config.sample_rate
        self.enable_automatic_punctuation = enable_automatic_punctuation if enable_automatic_punctuation is not None else config.enable_automatic_punctuation
        self.use_enhanced = use_enhanced if use_enhanced is not None else config.use_enhanced
        self.model = model or config.model
        
        # Initialize client
        self._init_client()
        
        # Log configuration
        logger.info(f"Initialized Google Cloud STT with model={self.model}, "
                    f"language={self.language_code}, enhanced={self.use_enhanced}")
    
    def _init_client(self):
        """Initialize the Google Cloud STT client with provided credentials."""
        try:
            # Check if credentials file exists
            if not os.path.exists(self.credentials_path):
                raise ValueError(f"Google Cloud credentials file not found: {self.credentials_path}")
            
            # Initialize credentials
            credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
            
            # Create the client
            self.client = speech.SpeechClient(credentials=credentials)
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud STT client: {e}")
            raise STTAPIError(f"Failed to initialize Google Cloud STT client: {str(e)}")
    
    def _get_recognition_config(self, streaming: bool = False, audio_sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """
        Get recognition configuration optimized for telephony.
        
        Args:
            streaming: Whether this is for streaming recognition
            audio_sample_rate: Override sample rate from audio file if provided
            
        Returns:
            Recognition configuration
        """
        # Use provided sample rate or default
        sample_rate = audio_sample_rate or self.sample_rate
        
        # Create configuration parameters
        config_params = {
            "language_code": self.language_code,
            "enable_automatic_punctuation": self.enable_automatic_punctuation,
        }
        
        # Add sample rate if provided
        if sample_rate:
            config_params["sample_rate_hertz"] = sample_rate
            
        # Add encoding
        config_params["encoding"] = speech.RecognitionConfig.AudioEncoding.LINEAR16
        
        # Add model if specified
        if self.model:
            config_params["model"] = self.model
            
        # Add use_enhanced if specified
        if self.use_enhanced:
            config_params["use_enhanced"] = self.use_enhanced
            
        # Add telephony-specific settings
        if self.model == "phone_call":
            # Add phone call-specific settings
            metadata = speech.RecognitionMetadata(
                interaction_type=speech.RecognitionMetadata.InteractionType.PHONE_CALL,
                microphone_distance=speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                original_media_type=speech.RecognitionMetadata.OriginalMediaType.AUDIO,
                recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.PHONE_LINE,
            )
            config_params["metadata"] = metadata
        
        # Add speech adaptation for better recognition of keywords
        if hasattr(config, 'speech_contexts') and config.speech_contexts:
            config_params["speech_contexts"] = [speech.SpeechContext(
                phrases=config.speech_contexts,
                boost=config.speech_context_boost
            )]
        
        return config_params
    
    async def transcribe(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Transcribe audio data in a single request.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Transcription result
        """
        try:
            # Try to detect sample rate from WAV header if present
            sample_rate = self._detect_wav_sample_rate(audio_data)
            
            # Run in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self._transcribe_sync(audio_data, sample_rate)
            )
        except Exception as e:
            logger.error(f"Error during STT transcription: {str(e)}")
            raise STTError(f"Google Cloud STT transcription error: {str(e)}")
    
    def _detect_wav_sample_rate(self, audio_data: bytes) -> Optional[int]:
        """
        Detect sample rate from WAV header if present.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Sample rate if detected, None otherwise
        """
        # Check if WAV header is present
        if len(audio_data) > 44 and audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            try:
                with io.BytesIO(audio_data) as wav_io:
                    with wave.open(wav_io, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        logger.info(f"Detected WAV sample rate: {sample_rate} Hz")
                        return sample_rate
            except Exception as e:
                logger.warning(f"Error reading WAV header: {e}")
        return None
    
    def _transcribe_sync(self, audio_data: bytes, sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """
        Synchronous method to transcribe speech using Google Cloud STT.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Override sample rate if provided
            
        Returns:
            Transcription result
        """
        try:
            # Create recognition config
            config_params = self._get_recognition_config(audio_sample_rate=sample_rate)
            recognition_config = speech.RecognitionConfig(**config_params)
            
            # Create recognition audio
            recognition_audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform transcription
            response = self.client.recognize(
                config=recognition_config,
                audio=recognition_audio
            )
            
            # Process results
            results = []
            for result in response.results:
                for alternative in result.alternatives:
                    word_info = []
                    if hasattr(alternative, 'words') and alternative.words:
                        for word in alternative.words:
                            word_info.append({
                                "word": word.word,
                                "start_time": word.start_time.total_seconds() if hasattr(word.start_time, 'total_seconds') else 0,
                                "end_time": word.end_time.total_seconds() if hasattr(word.end_time, 'total_seconds') else 0,
                                "confidence": word.confidence if hasattr(word, 'confidence') else 0
                            })
                    
                    results.append({
                        "transcript": alternative.transcript,
                        "confidence": alternative.confidence,
                        "words": word_info
                    })
            
            # Return the best result
            if results:
                best_result = max(results, key=lambda x: x["confidence"])
                return {
                    "transcription": best_result["transcript"],
                    "confidence": best_result["confidence"],
                    "words": best_result["words"],
                    "is_final": True
                }
            else:
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "words": [],
                    "is_final": True
                }
                
        except Exception as e:
            logger.error(f"Error in Google Cloud STT transcription: {e}")
            raise STTError(f"Google Cloud STT transcription error: {str(e)}")
    
    async def streaming_recognize(
        self,
        audio_generator: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream audio to Google Cloud STT and get real-time transcriptions.
        
        Args:
            audio_generator: Async generator producing audio chunks
            
        Yields:
            Transcription results as they become available
        """
        # Buffer for first chunk to detect sample rate
        first_chunk = None
        audio_chunks = []
        
        # Gather the first few chunks to detect sample rate if WAV
        async for chunk in audio_generator:
            if first_chunk is None:
                first_chunk = chunk
            audio_chunks.append(chunk)
            
            # Break after collecting some chunks
            if len(audio_chunks) >= 5:  # Collect 5 chunks max for initialization
                break
        
        # Detect sample rate from first chunk if it's WAV
        detected_sample_rate = None
        if first_chunk and len(first_chunk) > 44:
            if first_chunk[:4] == b'RIFF' and first_chunk[8:12] == b'WAVE':
                try:
                    with io.BytesIO(first_chunk) as wav_io:
                        with wave.open(wav_io, 'rb') as wav_file:
                            detected_sample_rate = wav_file.getframerate()
                            logger.info(f"Detected sample rate from first chunk: {detected_sample_rate} Hz")
                except Exception as e:
                    logger.warning(f"Error detecting sample rate from first chunk: {e}")
        
        # Get recognition config
        config_params = self._get_recognition_config(
            streaming=True, 
            audio_sample_rate=detected_sample_rate
        )
        
        # Create streaming config
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(**config_params),
            interim_results=True
        )
        
        # Create a thread-safe queue for passing audio chunks
        audio_queue = queue.Queue()
        
        # Add already collected chunks to the queue
        for chunk in audio_chunks:
            audio_queue.put(chunk)
        
        # Create a thread for collecting more audio chunks
        def audio_collector():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def collect_audio():
                nonlocal audio_generator
                try:
                    async for chunk in audio_generator:
                        audio_queue.put(chunk)
                    # Signal end of stream
                    audio_queue.put(None)
                except Exception as e:
                    logger.error(f"Error collecting audio: {e}")
                    audio_queue.put(None)  # Signal end on error
            
            loop.run_until_complete(collect_audio())
        
        # Start audio collection thread
        collector_thread = threading.Thread(target=audio_collector)
        collector_thread.daemon = True
        collector_thread.start()
        
        # Create generator for audio requests
        def request_generator():
            # First request must contain the config
            yield speech.StreamingRecognizeRequest(
                streaming_config=streaming_config
            )
            
            # Subsequent requests contain audio data
            while True:
                try:
                    chunk = audio_queue.get(block=True, timeout=10)
                    if chunk is None:  # End signal
                        break
                    
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                    audio_queue.task_done()
                except queue.Empty:
                    # Timeout waiting for audio
                    logger.warning("Timeout waiting for audio chunk")
                    break
                except Exception as e:
                    logger.error(f"Error in request generator: {e}")
                    break
        
        # Create a queue for passing recognition results back to the async context
        result_queue = asyncio.Queue()
        
        # Function to process responses in a separate thread
        def response_processor():
            try:
                # Start streaming recognition
                responses = self.client.streaming_recognize(request_generator())
                
                # Process each response
                for response in responses:
                    # Skip empty responses
                    if not response.results:
                        continue
                    
                    # Process each result
                    for result in response.results:
                        if not result.alternatives:
                            continue
                        
                        # Get best alternative
                        alternative = result.alternatives[0]
                        
                        # Create result dict
                        result_dict = {
                            "transcription": alternative.transcript,
                            "confidence": alternative.confidence if hasattr(alternative, "confidence") else 0.0,
                            "stability": result.stability if hasattr(result, "stability") else 1.0,
                            "is_final": result.is_final
                        }
                        
                        # Use asyncio.run to put the result in the queue from this thread
                        asyncio.run(result_queue.put(result_dict))
                
                # Signal end of results
                asyncio.run(result_queue.put(None))
                
            except Exception as e:
                logger.error(f"Error in response processor: {e}")
                # Signal error
                asyncio.run(result_queue.put({"error": str(e)}))
                asyncio.run(result_queue.put(None))
        
        # Start response processor thread
        processor_thread = threading.Thread(target=response_processor)
        processor_thread.daemon = True
        processor_thread.start()
        
        try:
            # Yield results as they become available
            while True:
                result = await result_queue.get()
                
                if result is None:  # End signal
                    break
                    
                # Check for error
                if "error" in result:
                    logger.error(f"Error in streaming recognition: {result['error']}")
                    break
                    
                yield result
                result_queue.task_done()
                
        finally:
            # Wait for threads to complete with timeout
            collector_thread.join(timeout=2)
            processor_thread.join(timeout=2)