"""
Deepgram Speech-to-Text client for batch processing.
"""
import os
import logging
import asyncio
import aiohttp
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, BinaryIO

from ..config import config
from .models import TranscriptionResult, TranscriptionConfig
from .exceptions import STTError, STTAPIError

logger = logging.getLogger(__name__)

class DeepgramSTT:
    """
    Client for the Deepgram Speech-to-Text API, optimized for telephony.
    
    This class handles batch STT operations using Deepgram's API,
    with configurations optimized for telephony voice applications.
    """
    
    BASE_URL = "https://api.deepgram.com/v1/listen"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        enable_caching: Optional[bool] = None,
    ):
        """
        Initialize the Deepgram STT client.
        
        Args:
            api_key: Deepgram API key (defaults to environment variable)
            model_name: STT model to use (defaults to config)
            language: Language code for recognition (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            enable_caching: Whether to cache results (defaults to config)
        """
        self.api_key = api_key or config.deepgram_api_key
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Set it in .env file or pass directly.")
        
        self.model_name = model_name or config.model_name
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path('./cache/stt_cache')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for Deepgram API requests."""
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/raw"
        }
    
    def _get_params(
        self, 
        config_obj: Optional[TranscriptionConfig] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get the parameters for an STT request.
        
        Args:
            config_obj: Optional configuration object
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Dictionary of parameters for the API request
        """
        # Start with model name and language
        params = {
            "model": self.model_name,
            "language": self.language,
        }
        
        # Add telephony-specific options for better accuracy
        params.update({
            "smart_format": config.smart_format,
            "filler_words": False,  # Filter out filler words
            "utterance_end_ms": str(config.utterance_end_ms),
            "alternatives": config.alternatives,
            "profanity_filter": config.profanity_filter,
            "diarize": config.diarize,
            "multichannel": config.multichannel,
            "tier": config.model_options.get("tier", "enhanced"),
        })
        
        # If we have keywords, add them
        if config.keywords:
            params["keywords"] = json.dumps(config.keywords)
        
        # Override with any config object settings
        if config_obj:
            params.update(config_obj.dict(exclude_none=True, exclude_unset=True))
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value
        
        return params
    
    def _get_cache_path(self, audio_bytes: bytes, params: Dict[str, Any]) -> Path:
        """
        Generate a cache file path based on audio content and parameters.
        
        Args:
            audio_bytes: Audio data
            params: Transcription parameters
            
        Returns:
            Path to cache file
        """
        # Create a unique hash based on audio and params
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        cache_key = f"{audio_hash}_{params_hash}"
        
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
        
        # Get parameters
        params = self._get_params(config_obj, **kwargs)
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(audio_bytes, params)
            if cache_path.exists():
                logger.debug(f"Found cached STT result for audio hash: {cache_path.stem}")
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    return TranscriptionResult(**cached_data)
                except Exception as e:
                    logger.warning(f"Error loading cached result: {e}, will transcribe again")
        
        # Make API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}",
                    headers=self._get_headers(),
                    params=params,
                    data=audio_bytes
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise STTAPIError(f"Deepgram API error: {response.status} - {error_msg}")
                    
                    response_data = await response.json()
            
            # Process results
            try:
                result = self._process_response(response_data)
                
                # Cache result if enabled
                if self.enable_caching:
                    with open(cache_path, 'w') as f:
                        json.dump(result.dict(), f)
                
                return result
            except KeyError as e:
                raise STTError(f"Unexpected response format: {e}")
            
        except aiohttp.ClientError as e:
            raise STTAPIError(f"Network error when connecting to Deepgram: {str(e)}")
        except Exception as e:
            raise STTError(f"Error during STT transcription: {str(e)}")
    
    def _process_response(self, response_data: Dict[str, Any]) -> TranscriptionResult:
        """
        Process Deepgram API response into a structured result.
        
        Args:
            response_data: Response data from Deepgram API
            
        Returns:
            Structured transcription result
        """
        # Extract the results
        results = response_data.get("results", {})
        channels = results.get("channels", [{}])
        
        # Get the best channel (usually only one for telephony)
        channel = channels[0]
        alternatives = channel.get("alternatives", [{}])
        
        # Get the best alternative
        if alternatives:
            best_alternative = alternatives[0]
            
            # Extract transcript and confidence
            transcript = best_alternative.get("transcript", "")
            confidence = best_alternative.get("confidence", 0.0)
            
            # Extract words
            words = best_alternative.get("words", [])
            
            # Get alternatives beyond the first one
            other_alternatives = [alt.get("transcript", "") for alt in alternatives[1:]]
            
            # Get audio duration
            metadata = results.get("metadata", {})
            audio_duration = metadata.get("duration", 0.0)
            
            return TranscriptionResult(
                text=transcript,
                confidence=confidence,
                words=words,
                alternatives=other_alternatives,
                audio_duration=audio_duration
            )
        else:
            # No alternatives found
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
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