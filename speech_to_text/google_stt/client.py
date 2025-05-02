"""
Google Cloud Speech-to-Text client for Voice AI Agent.
"""
import os
import logging
import asyncio
from typing import Optional, Dict, Any, List, AsyncIterator

from google.cloud import speech
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GoogleCloudSTT:
    """
    Client for Google Cloud Speech-to-Text API with streaming capabilities,
    optimized for telephony applications.
    """
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        language_code: str = "en-US",
        sample_rate: int = 8000,
        enable_automatic_punctuation: bool = True,
        use_enhanced: bool = True,
        model: str = "phone_call",
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
        self.credentials_path = credentials_path
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.use_enhanced = use_enhanced
        self.model = model
        
        # Initialize client
        self._init_client()
        
        logger.info(f"Initialized Google Cloud STT with model={model}, "
                   f"language={language_code}, enhanced={use_enhanced}")
    
    def _init_client(self):
        """Initialize the Google Cloud STT client with provided credentials."""
        try:
            # Check if credentials file is specified
            if self.credentials_path and os.path.exists(self.credentials_path):
                # Initialize credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                
                # Create the client
                self.client = speech.SpeechClient(credentials=credentials)
            else:
                # Use default credentials (environment variable)
                self.client = speech.SpeechClient()
                
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud STT client: {e}")
            raise
    
    def _get_recognition_config(self, streaming: bool = False) -> Dict[str, Any]:
        """
        Get recognition configuration optimized for telephony.
        
        Args:
            streaming: Whether this is for streaming recognition
            
        Returns:
            Recognition configuration
        """
        # Create configuration parameters for phone calls
        config_params = {
            "language_code": self.language_code,
            "enable_automatic_punctuation": self.enable_automatic_punctuation,
            "sample_rate_hertz": self.sample_rate,
            "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
        }
        
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
        speech_contexts = [
            "help", "agent", "cost", "price", "service", 
            "support", "information", "question", "connect"
        ]
        
        config_params["speech_contexts"] = [
            speech.SpeechContext(phrases=speech_contexts, boost=12.0)
        ]
        
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
            return await loop.run_in_executor(
                None, 
                lambda: self._transcribe_sync(audio_data)
            )
        except Exception as e:
            logger.error(f"Error during STT transcription: {str(e)}")
            raise
    
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
            raise
    
    async def streaming_recognize(
        self,
        audio_generator: AsyncIterator[bytes]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream audio to Google Cloud STT and get real-time transcriptions.
        
        Args:
            audio_generator: Async generator producing audio chunks
            
        Yields:
            Transcription results as they become available
        """
        # Create streaming config
        config_params = self._get_recognition_config(streaming=True)
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(**config_params),
            interim_results=True
        )
        
        # Create a generator for audio requests
        async def request_generator():
            # First request must contain the config
            yield speech.StreamingRecognizeRequest(
                streaming_config=streaming_config
            )
            
            # Subsequent requests contain audio data
            async for chunk in audio_generator:
                if not chunk:  # Skip empty chunks
                    continue
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
        
        # Process streaming response
        client = self.client
        
        # Run in a thread to avoid blocking
        requests = request_generator()
        responses = await self._run_streaming_recognize(client, requests)
        
        # Process streaming response
        async for response in responses:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                
                # Create result dict
                result_dict = {
                    "transcription": alternative.transcript,
                    "confidence": alternative.confidence if hasattr(alternative, "confidence") else 0.0,
                    "stability": result.stability if hasattr(result, "stability") else 1.0,
                    "is_final": result.is_final
                }
                
                yield result_dict
    
    async def _run_streaming_recognize(self, client, requests):
        """Run streaming recognition in a non-blocking way."""
        # Create a queue to pass results back
        result_queue = asyncio.Queue()
        
        # Define a thread function to process responses
        def process_responses():
            try:
                # Get responses from client
                responses_iterator = client.streaming_recognize(requests)
                
                # Push responses to queue
                for response in responses_iterator:
                    asyncio.run(result_queue.put(response))
                
                # Mark end of stream
                asyncio.run(result_queue.put(None))
            except Exception as e:
                logger.error(f"Error in streaming recognition: {e}")
                asyncio.run(result_queue.put(None))
        
        # Start the processing in a thread
        import threading
        thread = threading.Thread(target=process_responses)
        thread.daemon = True
        thread.start()
        
        # Yield results from the queue
        try:
            while True:
                response = await result_queue.get()
                if response is None:
                    break
                yield response
                result_queue.task_done()
        finally:
            # Wait for thread to complete with timeout
            thread.join(timeout=1.0)