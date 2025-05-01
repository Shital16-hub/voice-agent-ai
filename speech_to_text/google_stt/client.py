"""
Google Cloud Speech-to-Text client for batch processing.
"""
import os
import logging
import asyncio
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from google.cloud import speech
from google.cloud.speech import RecognitionConfig, RecognitionAudio
from google.oauth2 import service_account

from ..config import config
from .models import TranscriptionResult, TranscriptionConfig
from .exceptions import STTError, STTAPIError

logger = logging.getLogger(__name__)

class GoogleCloudSTT:
    """
    Client for the Google Cloud Speech-to-Text API, optimized for telephony.
    
    This class handles batch STT operations using Google Cloud's API,
    with configurations optimized for telephony voice applications.
    """
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None,
        language_code: Optional[str] = None,
        sample_rate: Optional[int] = None,
        enable_caching: Optional[bool] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Google Cloud STT client.
        
        Args:
            credentials_path: Path to Google credentials JSON (defaults to env variable)
            language_code: Language code (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            enable_caching: Whether to cache results (defaults to config)
            model: Model to use (defaults to phone_call)
        """
        # Use same credentials path as TTS if not provided
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not self.credentials_path:
            raise ValueError("Google Cloud credentials path is required. Set GOOGLE_APPLICATION_CREDENTIALS in .env or pass directly.")
        
        self.language_code = language_code or config.language
        self.sample_rate = sample_rate or 8000  # Default to 8kHz for telephony
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        self.model = model or "phone_call"
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path('./cache/stt_cache')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self._init_client()
        
        logger.info(f"Initialized Google Cloud STT client with model={self.model}, "
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
    
    def _get_recognition_config(self, config_obj: Optional[TranscriptionConfig] = None) -> Dict[str, Any]:
        """Get recognition configuration for transcription request."""
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
            "encoding": encoding_map.get("LINEAR16"),
            "audio_channel_count": 1,  # Mono for telephony
            "enable_automatic_punctuation": True,
            "model": self.model,
            "use_enhanced": True
        }
        
        # Add speech contexts for improved recognition
        speech_contexts = [{
            "phrases": ["price", "plan", "cost", "subscription", "service", "features", "support",
                       "help", "agent", "assistant", "voice", "stop", "continue"],
            "boost": 15.0
        }]
        recognition_config["speech_contexts"] = speech_contexts
        
        # Override with config object if provided
        if config_obj:
            config_dict = config_obj.dict(exclude_none=True, exclude_unset=True)
            for key, value in config_dict.items():
                if key == "encoding" and value in encoding_map:
                    recognition_config[key] = encoding_map[value]
                else:
                    recognition_config[key] = value
        
        return recognition_config
    
    def _get_cache_path(self, audio_bytes: bytes, config: Dict[str, Any]) -> Path:
        """
        Generate a cache file path based on audio content and parameters.
        
        Args:
            audio_bytes: Audio data
            config: Recognition configuration
            
        Returns:
            Path to cache file
        """
        # Create a unique hash based on audio and config
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        cache_key = f"{audio_hash}_{config_hash}"
        
        return self.cache_dir / f"{cache_key}.json"
    
    async def transcribe(
        self, 
        audio_bytes: bytes,
        config_obj: Optional[TranscriptionConfig] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_bytes: Audio data
            config_obj: Optional configuration object
            **kwargs: Additional parameters
            
        Returns:
            Transcription result
        """
        if not audio_bytes:
            logger.warning("Empty audio provided to transcribe")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
            )
        
        # Get recognition config
        recognition_config = self._get_recognition_config(config_obj)
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(audio_bytes, recognition_config)
            if cache_path.exists():
                logger.debug(f"Found cached STT result for audio hash: {cache_path.stem}")
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    return TranscriptionResult(**cached_data)
                except Exception as e:
                    logger.warning(f"Error loading cached result: {e}, will transcribe again")
        
        try:
            # Run in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            transcription_func = lambda: self._transcribe_sync(audio_bytes, recognition_config)
            result = await loop.run_in_executor(None, transcription_func)
            
            # Cache result if enabled
            if self.enable_caching:
                with open(cache_path, 'w') as f:
                    json.dump(result.dict(), f)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during STT transcription: {str(e)}")
            raise STTError(f"Google Cloud STT transcription error: {str(e)}")
    
    def _transcribe_sync(self, audio_bytes: bytes, recognition_config: Dict[str, Any]) -> TranscriptionResult:
        """
        Synchronous method to transcribe audio using Google Cloud STT.
        
        Args:
            audio_bytes: Audio data
            recognition_config: Recognition configuration
            
        Returns:
            Transcription result
        """
        try:
            # Create RecognitionAudio object
            audio = RecognitionAudio(content=audio_bytes)
            
            # Create RecognitionConfig object
            config = RecognitionConfig(**recognition_config)
            
            # Perform synchronous recognition
            response = self.client.recognize(config=config, audio=audio)
            
            # Process results
            return self._process_response(response)
            
        except Exception as e:
            logger.error(f"Error in Google Cloud STT transcription: {e}")
            raise STTError(f"Google Cloud STT transcription error: {str(e)}")
    
    def _process_response(self, response) -> TranscriptionResult:
        """
        Process Google Cloud API response into a structured result.
        
        Args:
            response: Response from Google Cloud STT API
            
        Returns:
            Structured transcription result
        """
        # Check if we have any results
        if not response.results:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
            )
        
        # Get the best alternative from the first result
        result = response.results[0]
        
        if not result.alternatives:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
            )
        
        # Get the first alternative
        alternative = result.alternatives[0]
        transcript = alternative.transcript
        confidence = alternative.confidence
        
        # Extract word information if available
        words = []
        if hasattr(alternative, 'words') and alternative.words:
            for word_info in alternative.words:
                word = {
                    "word": word_info.word,
                    "start_time": word_info.start_time.total_seconds(),
                    "end_time": word_info.end_time.total_seconds(),
                    "confidence": confidence
                }
                words.append(word)
        
        # Get other alternatives
        other_alternatives = []
        if len(result.alternatives) > 1:
            other_alternatives = [alt.transcript for alt in result.alternatives[1:]]
        
        # Estimate audio duration from words or results
        audio_duration = 0.0
        if words and len(words) > 0:
            last_word = words[-1]
            audio_duration = last_word.get("end_time", 0.0)
        
        return TranscriptionResult(
            text=transcript,
            confidence=confidence,
            words=words,
            alternatives=other_alternatives,
            audio_duration=audio_duration
        )
    
    async def transcribe_file(
        self, 
        file_path: str,
        config_obj: Optional[TranscriptionConfig] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            config_obj: Optional configuration object
            **kwargs: Additional parameters
            
        Returns:
            Transcription result
        """
        try:
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            
            return await self.transcribe(audio_bytes, config_obj, **kwargs)
        except FileNotFoundError:
            raise STTError(f"File not found: {file_path}")
        except Exception as e:
            raise STTError(f"Error transcribing file: {str(e)}")