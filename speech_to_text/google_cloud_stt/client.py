# speech_to_text/google_cloud_stt/client.py
"""
Google Cloud Speech-to-Text client for batch processing.
"""
import os
import logging
from typing import Dict, Any, Optional, Union, List
import time

# Google Cloud Speech imports
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import types
from google.oauth2 import service_account

from ..config import config
from .models import TranscriptionResult, TranscriptionConfig
from .exceptions import STTError, STTAPIError

logger = logging.getLogger(__name__)

class GoogleCloudSTT:
    """
    Client for Google Cloud Speech-to-Text API, optimized for telephony.
    
    This class handles batch STT operations using Google Cloud Speech API,
    with configurations optimized for telephony voice applications.
    """
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        enable_caching: Optional[bool] = None,
    ):
        """
        Initialize the Google Cloud STT client.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            language: Language code for recognition (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            enable_caching: Whether to cache results (defaults to config)
        """
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        
        # Set up credentials
        self._setup_credentials()
        
        # Initialize client
        self.client = speech.SpeechClient(credentials=self.credentials)
        
        logger.info(f"Initialized Google Cloud STT client with language: {self.language}")
    
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

    def _get_speech_config(self, config_obj: Optional[TranscriptionConfig] = None, **kwargs) -> Dict[str, Any]:
        """
        Get the speech recognition configuration.
        
        Args:
            config_obj: Optional configuration object
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Dictionary with recognition config parameters
        """
        # Create recognition config with telephony optimizations
        recognition_config = {
            "language_code": self.language,
            "sample_rate_hertz": self.sample_rate,
            "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "model": "phone_call",  # Use phone_call model for telephony
            "use_enhanced": True,  # Use enhanced model
            "audio_channel_count": 1,  # Mono audio
            "enable_separate_recognition_per_channel": False,
            # Noise reduction settings
            "enable_automatic_punctuation": True,
            # Configure for telephony
            "speech_contexts": [{
                "phrases": config.keywords,  # Business-related phrases to boost
                "boost": 10.0  # Boost factor for these phrases
            }],
        }
        
        # Apply settings for better noise handling
        if config.enable_noise_filtering:
            recognition_config.update({
                "use_enhanced": True,
                "model": "phone_call",
            })
        
        # Add telephony-specific options
        recognition_config.update({
            # Telephony-related settings
            "diarization_config": {
                "enable_speaker_diarization": False,  # Single speaker in telephony context
            },
        })
        
        # Process any overrides from config object
        if config_obj:
            for key, value in config_obj.dict(exclude_none=True, exclude_unset=True).items():
                if key in recognition_config and value is not None:
                    recognition_config[key] = value
        
        # Apply any additional keyword arguments
        for key, value in kwargs.items():
            if value is not None:
                recognition_config[key] = value
        
        return recognition_config
    
    async def transcribe(
        self, 
        audio_data: bytes,
        config_obj: Optional[TranscriptionConfig] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as bytes
            config_obj: Optional configuration object
            **kwargs: Additional parameters
            
        Returns:
            Transcription result
        """
        if not audio_data:
            logger.warning("Empty audio provided to transcribe")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
            )
        
        try:
            # Get recognition config
            recognition_config = self._get_speech_config(config_obj, **kwargs)
            
            # Create proper config object
            config = speech.RecognitionConfig(**recognition_config)
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform synchronous speech recognition
            start_time = time.time()
            response = self.client.recognize(
                config=config,
                audio=audio
            )
            processing_time = time.time() - start_time
            
            logger.info(f"Transcribed {len(audio_data)} bytes in {processing_time:.2f}s")
            
            # Process results
            return self._process_response(response)
        
        except Exception as e:
            logger.error(f"Error in Google Cloud transcription: {e}")
            raise STTError(f"Error during STT transcription: {str(e)}")
    
    def _process_response(self, response) -> TranscriptionResult:
        """
        Process Google Cloud Speech API response.
        
        Args:
            response: Google Cloud Speech response
            
        Returns:
            Structured transcription result
        """
        if not response.results:
            logger.warning("No transcription results returned")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
            )
        
        # Get the first result (highest confidence)
        result = response.results[0]
        
        if not result.alternatives:
            logger.warning("No alternatives in transcription result")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
            )
        
        # Get the first alternative (highest confidence)
        alternative = result.alternatives[0]
        
        # Extract transcript and confidence
        transcript = alternative.transcript
        confidence = alternative.confidence
        
        # Extract words with timing info if available
        words = []
        if hasattr(alternative, 'words') and alternative.words:
            for word_info in alternative.words:
                words.append({
                    "word": word_info.word,
                    "start_time": word_info.start_time.total_seconds(),
                    "end_time": word_info.end_time.total_seconds(),
                    "confidence": confidence  # Per-word confidence not provided
                })
        
        # Get alternatives beyond the first one
        other_alternatives = []
        for alt in result.alternatives[1:]:
            other_alternatives.append(alt.transcript)
        
        # Calculate audio duration based on word timings if available
        audio_duration = 0.0
        if words:
            audio_duration = words[-1]["end_time"]
        
        return TranscriptionResult(
            text=transcript,
            confidence=confidence,
            words=words,
            alternatives=other_alternatives,
            audio_duration=audio_duration
        )