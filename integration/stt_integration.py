"""
Speech-to-Text integration for Voice AI Agent using Google Cloud STT.
"""
import logging
import time
import asyncio
import re
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator
import numpy as np

from speech_to_text.google_stt import GoogleCloudSTT, STTStreamer, SpeechDetector
from speech_to_text.utils.audio_utils import load_audio_file

logger = logging.getLogger(__name__)

# Patterns for non-speech annotations
NON_SPEECH_PATTERNS = [
    r'\[.*?\]',           # Anything in square brackets
    r'\(.*?\)',           # Anything in parentheses
    r'\<.*?\>',           # Anything in angle brackets
]

class STTIntegration:
    """
    Speech-to-Text integration for Voice AI Agent.
    
    Provides an abstraction layer for Google Cloud STT functionality,
    handling audio processing and transcription with optimizations for telephony.
    """
    
    def __init__(
        self,
        stt_client: Optional[GoogleCloudSTT] = None,
        language: str = "en"
    ):
        """
        Initialize the STT integration.
        
        Args:
            stt_client: Initialized GoogleCloudSTT instance
            language: Language code for speech recognition
        """
        self.stt_client = stt_client
        self.language = language
        self.initialized = True if stt_client else False
        
        # Initialize streamer and speech detector
        self.streamer = None
        self.speech_detector = None
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Minimum word count for valid queries
        self.min_words_for_valid_query = 3
    
    async def init(self, credentials_path: Optional[str] = None) -> None:
        """
        Initialize the STT component if not already initialized.
        
        Args:
            credentials_path: Path to Google credentials file
        """
        if self.initialized:
            return
            
        try:
            # Create Google Cloud STT client
            self.stt_client = GoogleCloudSTT(
                credentials_path=credentials_path,
                language_code=self.language
            )
            
            # Create streamer
            self.streamer = STTStreamer(stt_client=self.stt_client)
            
            # Create speech detector
            self.speech_detector = SpeechDetector()
            
            self.initialized = True
            logger.info(f"Initialized STT with Google Cloud STT and language: {self.language}")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Clean up transcription text by removing non-speech annotations.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
            
        # Remove non-speech annotations
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove common filler words at beginning of sentences
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Log what was cleaned if substantial
        if text != cleaned_text:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Check if a transcription is valid and worth processing.
        
        Args:
            text: Transcription text
            
        Returns:
            True if the transcription is valid
        """
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if it's empty after cleaning
        if not cleaned_text:
            logger.info("Transcription contains only non-speech annotations")
            return False
        
        # Add noise keyword filtering
        noise_keywords = ["click", "keyboard", "tongue", "scoff", "static", "*"]
        if any(keyword in text.lower() for keyword in noise_keywords):
            logger.info(f"Transcription contains noise keywords: {text}")
            return False
        
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_valid_query:
            logger.info(f"Transcription too short: {word_count} words")
            return False
        
        return True
    
    async def transcribe_audio_file(
        self,
        audio_file_path: str,
        callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_file_path: Path to audio file
            callback: Optional callback for results
            
        Returns:
            Dictionary with transcription results
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Load audio file
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=8000)
            audio_duration = len(audio) / sample_rate
            
            # Convert to bytes (16-bit PCM)
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Transcribe
            result = await self.stt_client.transcribe(audio_bytes)
            
            # Clean up the transcription
            transcription = result.get("transcription", "")
            cleaned_text = self.cleanup_transcription(transcription)
            
            # Add processing time
            result["processing_time"] = time.time() - start_time
            result["duration"] = audio_duration
            result["transcription"] = cleaned_text
            result["original_transcription"] = transcription
            result["is_valid"] = self.is_valid_transcription(cleaned_text)
            
            # Call callback if provided
            if callback and result["is_valid"]:
                await callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def transcribe_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray, List[float]],
        callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            callback: Optional callback for results
            
        Returns:
            Dictionary with transcription results
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Convert to bytes if needed
            if isinstance(audio_data, np.ndarray):
                audio_data = (audio_data * 32767).astype(np.int16).tobytes()
            elif isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
                audio_data = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Transcribe
            result = await self.stt_client.transcribe(audio_data)
            
            # Clean up the transcription
            transcription = result.get("transcription", "")
            cleaned_text = self.cleanup_transcription(transcription)
            
            # Add processing time
            result["processing_time"] = time.time() - start_time
            result["transcription"] = cleaned_text
            result["original_transcription"] = transcription
            result["is_valid"] = self.is_valid_transcription(cleaned_text)
            
            # Detect speech in audio
            result["contains_speech"] = self.speech_detector.detect_speech(
                audio_data, result
            ) if self.speech_detector else (cleaned_text != "")
            
            # Call callback if provided
            if callback and result["is_valid"]:
                await callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def start_streaming(self) -> None:
        """Start a new streaming transcription session."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return
        
        if not self.streamer:
            self.streamer = STTStreamer(stt_client=self.stt_client)
        
        await self.streamer.start_streaming()
        logger.debug("Started streaming transcription session")
    
    async def process_stream_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray, List[float]],
        callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a chunk of streaming audio.
        
        Args:
            audio_chunk: Audio chunk data
            callback: Optional callback for results
            
        Returns:
            Transcription result or None for interim results
        """
        if not self.initialized or not self.streamer:
            logger.error("STT integration not properly initialized")
            return None
        
        # Convert to bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()
        elif isinstance(audio_chunk, list):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
            audio_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()
        
        # Process through streamer
        result = await self.streamer.process_audio_chunk(audio_chunk)
        
        # Process result if available
        if result:
            # Clean up transcription
            transcription = result.get("transcription", "")
            cleaned_text = self.cleanup_transcription(transcription)
            
            # Update result
            result["transcription"] = cleaned_text
            result["original_transcription"] = transcription
            result["is_valid"] = self.is_valid_transcription(cleaned_text)
            
            # Check for speech
            result["contains_speech"] = self.speech_detector.detect_speech(
                audio_chunk, result
            ) if self.speech_detector else (cleaned_text != "")
            
            # Check for barge-in
            result["barge_in_detected"] = self.speech_detector.check_for_barge_in(
                audio_chunk, result
            ) if self.speech_detector else False
            
            # Call callback if provided
            if callback and result["is_valid"]:
                await callback(result)
        
        return result
    
    async def end_streaming(self) -> Tuple[str, float]:
        """
        End the streaming session and get final transcription.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.initialized or not self.streamer:
            logger.error("STT integration not properly initialized")
            return "", 0.0
        
        # Stop streaming and get final result
        result = await self.streamer.stop_streaming()
        
        if result:
            # Clean up transcription
            transcription = result.get("transcription", "")
            cleaned_text = self.cleanup_transcription(transcription)
            
            # Get duration if available
            duration = 0.0
            if "words" in result and result["words"]:
                words = result["words"]
                if words and "end_time" in words[-1]:
                    duration = words[-1]["end_time"]
            
            return cleaned_text, duration
        
        return "", 0.0
    
    def set_agent_speaking(self, is_speaking: bool) -> None:
        """
        Update the agent speaking state for barge-in detection.
        
        Args:
            is_speaking: Whether the agent is currently speaking
        """
        if self.speech_detector:
            self.speech_detector.set_agent_speaking(is_speaking)