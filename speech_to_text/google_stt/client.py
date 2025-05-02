"""
Google Cloud Speech-to-Text client for Voice AI Agent.
"""
import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
import io

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
    
    def _get_recognition_config(self, streaming: bool = False) -> Dict[str, Any]:
        """
        Get recognition configuration optimized for telephony.
        
        Args:
            streaming: Whether this is for streaming recognition
            
        Returns:
            Recognition configuration
        """
        # Updated for v2.x of the API - uses direct enum values instead of importing enums
        config_params = {
            "language_code": self.language_code,
            "sample_rate_hertz": self.sample_rate,
            "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "model": self.model,
            "use_enhanced": self.use_enhanced,
            "enable_automatic_punctuation": self.enable_automatic_punctuation,
        }
        
        # Add telephony-specific settings
        if self.model == "phone_call":
            # Add phone call-specific settings - updated for v2.x
            config_params["metadata"] = speech.RecognitionMetadata(
                interaction_type=speech.RecognitionMetadata.InteractionType.PHONE_CALL,
                microphone_distance=speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                original_media_type=speech.RecognitionMetadata.OriginalMediaType.AUDIO,
                recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.PHONE_LINE,
            )
        
        # Add speech adaptation for better recognition of keywords
        if config.speech_contexts:
            config_params["speech_contexts"] = [speech.SpeechContext(
                phrases=config.speech_contexts,
                boost=config.speech_context_boost
            )]
        
        # Add streaming-specific settings
        if streaming:
            config_params["interim_results"] = True
        
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
            # Run in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._transcribe_sync, audio_data)
        except Exception as e:
            logger.error(f"Error during STT transcription: {str(e)}")
            raise STTError(f"Google Cloud STT transcription error: {str(e)}")
    
    def _transcribe_sync(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Synchronous method to transcribe speech using Google Cloud STT.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Transcription result
        """
        try:
            # Create recognition config
            config_params = self._get_recognition_config()
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
        # Set up the streaming config
        config_params = self._get_recognition_config(streaming=True)
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(**config_params),
            interim_results=True
        )
        
        # Create an async queue for results
        result_queue = asyncio.Queue()
        
        # Function to run in a separate thread
        def process_stream():
            requests = []
            
            # First request must contain config but no audio
            requests.append(speech.StreamingRecognizeRequest(
                streaming_config=streaming_config
            ))
            
            # Create generator for audio data
            def audio_request_generator():
                # Yield the config request first
                yield requests[0]
                
                # Then yield audio data requests
                while True:
                    try:
                        audio_chunk = audio_chunks_queue.get(block=True, timeout=10)
                        if audio_chunk is None:  # End signal
                            break
                        
                        yield speech.StreamingRecognizeRequest(
                            audio_content=audio_chunk
                        )
                    except Exception as e:
                        logger.error(f"Error in audio request generator: {e}")
                        break
            
            try:
                # Start streaming recognition
                responses = self.client.streaming_recognize(
                    requests=audio_request_generator()
                )
                
                # Process responses
                for response in responses:
                    # Put response in queue for async processing
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put(response),
                        asyncio.get_event_loop()
                    )
            except Exception as e:
                logger.error(f"Error in streaming recognition: {e}")
                # Put None to signal error
                asyncio.run_coroutine_threadsafe(
                    result_queue.put(None),
                    asyncio.get_event_loop()
                )
        
        # Create queue for audio chunks
        audio_chunks_queue = asyncio.Queue()
        
        # Start processing thread
        import threading
        process_thread = threading.Thread(target=process_stream)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            # Gather audio chunks
            async for audio_chunk in audio_generator:
                await audio_chunks_queue.put(audio_chunk)
            
            # Signal end of audio
            await audio_chunks_queue.put(None)
            
            # Process results
            while True:
                try:
                    response = await asyncio.wait_for(result_queue.get(), timeout=10)
                    if response is None:  # Error signal
                        break
                    
                    # Process streaming response
                    for result in response.results:
                        yield {
                            "transcription": result.alternatives[0].transcript if result.alternatives else "",
                            "confidence": result.alternatives[0].confidence if result.alternatives else 0.0,
                            "stability": result.stability if hasattr(result, 'stability') else 1.0,
                            "is_final": result.is_final
                        }
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for transcription results")
                    break
                except Exception as e:
                    logger.error(f"Error processing transcription result: {e}")
                    break
        finally:
            # Clean up
            if process_thread.is_alive():
                process_thread.join(timeout=2)