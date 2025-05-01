"""
Speech-to-Text integration module for Voice AI Agent with Google Cloud Speech-to-Text.

This module provides classes and functions for integrating Google Cloud speech-to-text
capabilities with the Voice AI Agent system.
"""
import logging
import time
import asyncio
import re
import numpy as np
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator

from speech_to_text.google_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from speech_to_text.utils.audio_utils import load_audio_file

logger = logging.getLogger(__name__)

# Enhanced patterns for non-speech annotations
NON_SPEECH_PATTERNS = [
    r'\[.*?\]',           # Anything in square brackets
    r'\(.*?\)',           # Anything in parentheses
    r'\<.*?\>',           # Anything in angle brackets
    r'music playing',     # Common transcription
    r'background noise',  # Common transcription
    r'static',            # Common transcription
    r'\b(um|uh|hmm|mmm)\b',  # Common filler words
    # Add specific telephony noise patterns
    r'phone ringing',     # Phone ringing sound
    r'dial tone',         # Dial tone
    r'busy signal',       # Busy signal
    r'beep',              # Beep sounds
]

class STTIntegration:
    """
    Speech-to-Text integration for Voice AI Agent with Google Cloud Speech-to-Text.
    
    Provides an abstraction layer for speech recognition functionality,
    handling audio processing and transcription with improved speech/noise discrimination.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamingSTT] = None,
        language: str = "en-US"
    ):
        """
        Initialize the STT integration.
        
        Args:
            speech_recognizer: Initialized GoogleCloudStreamingSTT instance
            language: Language code for speech recognition
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Enhanced adaptive noise floor tracking
        self.noise_samples = []
        self.max_samples = 30  # Increased for better statistics
        self.ambient_noise_level = 0.015  # Starting threshold
        self.min_noise_floor = 0.008  # Minimum noise floor
        
        # Track minimum words for valid query
        self.min_words_for_valid_query = 3  # Increased from 2
    
    async def init(self, credentials_path: Optional[str] = None) -> None:
        """
        Initialize the STT component if not already initialized.
        
        Args:
            credentials_path: Path to Google credentials (optional)
        """
        if self.initialized:
            return
            
        try:
            # Create a new Google Speech-to-Text streaming client optimized for telephony
            self.speech_recognizer = GoogleCloudStreamingSTT(
                credentials_path=credentials_path,
                language_code=self.language,
                sample_rate=8000,  # 8kHz for telephony
                encoding="LINEAR16",
                channels=1,
                interim_results=True,
                model="phone_call"  # Use phone_call model for better telephony results
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with Google Cloud Speech-to-Text and language: {self.language}")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def cleanup_transcription(self, text: str) -> str:
        """
        More lenient cleanup of transcription text.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
            
        # Keep simple cleanup but reduce filtering
        cleaned_text = text
        
        # Only remove obvious noise markers
        cleaned_text = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Log what was cleaned if substantial
        if text != cleaned_text:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text
    
    def is_valid_transcription(self, text: str, min_words: int = None) -> bool:
        """
        More lenient check if a transcription is valid and worth processing.
        
        Args:
            text: Transcription text
            min_words: Minimum word count to be considered valid
            
        Returns:
            True if the transcription is valid
        """
        if min_words is None:
            min_words = 2  # Reduced from 3
            
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if it's empty after cleaning
        if not cleaned_text:
            logger.info("Transcription contains only non-speech annotations")
            return False
        
        # Reduced filtering - only filter extreme noise cases
        extreme_noise_keywords = ["static", "(", ")", "[", "]"]
        if any(keyword in text.lower() for keyword in extreme_noise_keywords) and len(cleaned_text.split()) <= 1:
            logger.info(f"Filtered extreme noise transcription: '{text}'")
            return False
        
        # Check word count - more lenient
        word_count = len(cleaned_text.split())
        if word_count < min_words:
            logger.info(f"Transcription too short: {word_count} words")
            return False
            
        return True
    
    async def transcribe_audio_file(
        self,
        audio_file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file with enhanced speech processing.
        
        Args:
            audio_file_path: Path to audio file
            callback: Optional callback for interim results
            
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
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=8000)  # 8kHz for telephony
            audio_duration = len(audio) / sample_rate
            
            # Convert to bytes (16-bit PCM)
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Get results
            final_results = []
            
            # Define a custom callback to store results
            async def store_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                
                # Call the original callback if provided
                if callback:
                    await callback(result)
            
            # Start streaming session
            await self.speech_recognizer.start_streaming()
            
            # Process the audio in chunks
            chunk_size = 4096  # Approximately 256ms of audio at 8kHz
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                await self.speech_recognizer.process_audio_chunk(chunk, store_result)
            
            # Stop streaming to get final results
            final_text, _ = await self.speech_recognizer.stop_streaming()
            
            # Combine results if we have any
            if final_results:
                # Use the best final result (with highest confidence)
                best_result = max(final_results, key=lambda r: r.confidence if hasattr(r, 'confidence') else 0.0)
                
                # Clean up the transcription
                cleaned_text = self.cleanup_transcription(best_result.text)
                
                return {
                    "transcription": cleaned_text,
                    "original_transcription": best_result.text,
                    "confidence": best_result.confidence if hasattr(best_result, 'confidence') else 1.0,
                    "duration": audio_duration,
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": self.is_valid_transcription(cleaned_text)
                }
            elif final_text:
                # Use final text from stop_streaming if no results collected
                cleaned_text = self.cleanup_transcription(final_text)
                
                return {
                    "transcription": cleaned_text,
                    "original_transcription": final_text,
                    "confidence": 1.0,  # Default confidence
                    "duration": audio_duration,
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": self.is_valid_transcription(cleaned_text)
                }
            else:
                logger.warning("No transcription results obtained")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": audio_duration,
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": False
                }
            
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def transcribe_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray, List[float]],
        is_short_audio: bool = False,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio data with Google Cloud Speech-to-Text.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            is_short_audio: Flag to indicate short audio for optimized handling
            callback: Optional callback for interim results
            
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
                # Ensure 16-bit PCM format
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            elif isinstance(audio_data, list):
                # Convert list to numpy array, then to bytes
                audio_array = np.array(audio_data, dtype=np.float32)
                audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            else:
                # Already bytes
                audio_bytes = audio_data
            
            # Get results
            final_results = []
            
            # Define a custom callback to store results
            async def store_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                
                # Call the original callback if provided
                if callback:
                    await callback(result)
            
            # Start streaming session if needed
            if not getattr(self.speech_recognizer, 'is_streaming', False):
                await self.speech_recognizer.start_streaming()
            
            # Process the audio in chunks
            chunk_size = 4096  # For telephony audio (smaller chunk size)
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                result = await self.speech_recognizer.process_audio_chunk(chunk, store_result)
                # Note: result here is the return value from process_audio_chunk
                if result and result.is_final and not result in final_results:
                    final_results.append(result)
            
            # Combine results if we have any
            if final_results:
                # Use the best final result (with highest confidence)
                best_result = max(final_results, key=lambda r: r.confidence if hasattr(r, 'confidence') else 0.0)
                
                # Clean up the transcription
                cleaned_text = self.cleanup_transcription(best_result.text)
                
                return {
                    "transcription": cleaned_text,
                    "original_transcription": best_result.text,
                    "confidence": best_result.confidence if hasattr(best_result, 'confidence') else 1.0,
                    "duration": len(audio_bytes) / (2 * 8000),  # Estimate duration (16-bit, 8kHz)
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": self.is_valid_transcription(cleaned_text)
                }
            else:
                # Get whatever might be the last processed result
                last_result = getattr(self.speech_recognizer, 'last_result', None)
                if last_result and hasattr(last_result, 'text'):
                    cleaned_text = self.cleanup_transcription(last_result.text)
                    
                    return {
                        "transcription": cleaned_text,
                        "original_transcription": last_result.text,
                        "confidence": last_result.confidence if hasattr(last_result, 'confidence') else 0.0,
                        "duration": len(audio_bytes) / (2 * 8000),  # Estimate duration
                        "processing_time": time.time() - start_time,
                        "is_final": True,
                        "is_valid": self.is_valid_transcription(cleaned_text)
                    }
                
                logger.warning("No transcription results obtained")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": len(audio_bytes) / (2 * 8000),  # Estimate duration
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": False
                }
            
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
        
        await self.speech_recognizer.start_streaming()
        logger.debug("Started streaming transcription session")
    
    async def process_stream_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray, List[float]],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a chunk of streaming audio.
        
        Args:
            audio_chunk: Audio chunk data
            callback: Optional callback for results
            
        Returns:
            Transcription result or None for interim results
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return None
        
        # Convert to bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            # Ensure 16-bit PCM format
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
        elif isinstance(audio_chunk, list):
            # Convert list to numpy array, then to bytes
            audio_array = np.array(audio_chunk, dtype=np.float32)
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        else:
            # Already bytes
            audio_bytes = audio_chunk
        
        # Create a custom callback to clean up transcriptions
        async def clean_callback(result: StreamingTranscriptionResult):
            if result and hasattr(result, 'text') and result.text:
                # Store original for debugging
                original_text = result.text
                
                # Clean up the text
                result.text = self.cleanup_transcription(result.text)
                
                # Log the change if significant
                if original_text != result.text:
                    logger.debug(f"Cleaned interim transcription: '{original_text}' -> '{result.text}'")
                
                # Only call user callback for valid transcriptions
                if result.text and callback:
                    await callback(result)
        
        # Process the audio chunk
        return await self.speech_recognizer.process_audio_chunk(
            audio_chunk=audio_bytes,
            callback=clean_callback
        )
    
    async def end_streaming(self) -> Tuple[str, float]:
        """
        End the streaming session and get final transcription.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return "", 0.0
        
        # Stop streaming session
        result = await self.speech_recognizer.stop_streaming()
        
        # Check if we got a tuple or other return value
        if isinstance(result, tuple) and len(result) >= 2:
            text, duration = result
        else:
            # Handle case where stop_streaming returns something else (e.g., None)
            text = ""
            duration = 0.0
            # Check if last_result is available
            last_result = getattr(self.speech_recognizer, 'last_result', None)
            if last_result and hasattr(last_result, 'text'):
                text = last_result.text
                duration = last_result.end_time - last_result.start_time if hasattr(last_result, 'start_time') and hasattr(last_result, 'end_time') else 0.0
        
        # Clean up the transcription
        cleaned_text = self.cleanup_transcription(text)
        
        # Log what was changed if significant
        if text != cleaned_text:
            logger.info(f"Cleaned final transcription: '{text}' -> '{cleaned_text}'")
        
        return cleaned_text, duration
    
    async def process_realtime_audio_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None,
        silence_frames_threshold: int = 30
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process real-time audio stream and detect utterances.
        
        Args:
            audio_stream: Async iterator yielding audio chunks
            callback: Optional callback for interim results
            silence_frames_threshold: Number of silence frames to consider end of utterance
            
        Yields:
            Transcription results for each detected utterance
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            yield {"error": "STT integration not initialized"}
            return
        
        # Start streaming
        await self.speech_recognizer.start_streaming()
        
        # Track consecutive silence frames
        silence_frames = 0
        max_silence_frames = silence_frames_threshold
        
        try:
            async for audio_chunk in audio_stream:
                # Convert to bytes (16-bit PCM)
                if isinstance(audio_chunk, np.ndarray):
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = audio_chunk
                
                # Define a result processing callback
                results = []
                
                async def process_result(result: StreamingTranscriptionResult):
                    # Store the result
                    results.append(result)
                    
                    # Clean up the text
                    if hasattr(result, 'text') and result.text:
                        result.text = self.cleanup_transcription(result.text)
                    
                    # Call the original callback if provided and valid
                    if callback and result.text and self.is_valid_transcription(result.text):
                        await callback(result)
                
                # Process the audio chunk
                await self.speech_recognizer.process_audio_chunk(
                    audio_chunk=audio_bytes,
                    callback=process_result
                )
                
                # Get final results (if any)
                final_results = [r for r in results if r.is_final]
                
                # If we have final results, yield them
                for result in final_results:
                    cleaned_text = self.cleanup_transcription(result.text)
                    
                    # Only yield if it's a valid transcription after cleaning
                    if cleaned_text and self.is_valid_transcription(cleaned_text):
                        logger.info(f"Utterance detected: {cleaned_text}")
                        
                        # Yield the result
                        yield {
                            "transcription": cleaned_text,
                            "original_transcription": result.text,
                            "duration": result.end_time - result.start_time if hasattr(result, 'start_time') else 0.0,
                            "is_final": True,
                            "is_valid": True
                        }
                
                # Check for silence based on results being empty
                if not results:
                    silence_frames += 1
                else:
                    silence_frames = 0
                
                # If we've detected enough silence, check for any final results
                if silence_frames >= max_silence_frames:
                    final_text, duration = await self.end_streaming()
                    
                    # Clean up the transcription
                    cleaned_text = self.cleanup_transcription(final_text)
                    
                    # Only yield if it's a valid transcription after cleaning
                    if cleaned_text and self.is_valid_transcription(cleaned_text):
                        logger.info(f"Final utterance detected: {cleaned_text}")
                        
                        # Yield the result
                        yield {
                            "transcription": cleaned_text,
                            "original_transcription": final_text,
                            "duration": duration,
                            "is_final": True,
                            "is_valid": True
                        }
                    
                    # Reset for next utterance
                    await self.speech_recognizer.start_streaming()
                    silence_frames = 0
        
        except Exception as e:
            logger.error(f"Error in real-time audio processing: {e}")
            yield {"error": str(e)}
        
        finally:
            # Clean up
            await self.speech_recognizer.stop_streaming()