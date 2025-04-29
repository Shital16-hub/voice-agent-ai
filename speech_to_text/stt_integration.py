"""
Enhanced Speech-to-Text integration module for Voice AI Agent with improved noise handling.

This module provides classes and functions for integrating Deepgram speech-to-text
capabilities with the Voice AI Agent system.
"""
import logging
import time
import asyncio
import re
import numpy as np
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator
from scipy import signal

from speech_to_text.deepgram_stt import DeepgramStreamingSTT, StreamingTranscriptionResult
from speech_to_text.utils.audio_utils import load_audio_file

logger = logging.getLogger(__name__)

# Define expanded patterns for non-speech annotations
NON_SPEECH_PATTERNS = [
    r'\[.*?\]',           # Anything in square brackets
    r'\(.*?\)',           # Anything in parentheses
    r'\<.*?\>',           # Anything in angle brackets
    r'music playing',     # Common transcription
    r'background noise',  # Common transcription
    r'static',            # Common transcription
    r'\b(um|uh|hmm|mmm)\b',  # Common filler words
]

class STTIntegration:
    """
    Enhanced Speech-to-Text integration for Voice AI Agent with Deepgram.
    
    Provides an abstraction layer for speech recognition functionality,
    handling audio processing and transcription.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[DeepgramStreamingSTT] = None,
        language: str = "en"
    ):
        """
        Initialize the STT integration.
        
        Args:
            speech_recognizer: Initialized DeepgramStreamingSTT instance
            language: Language code for speech recognition
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Keep track of average audio levels for adaptive thresholding
        self.noise_samples = []
        self.speech_samples = []
        self.max_samples = 20
        self.ambient_noise_level = 0.01  # Starting threshold
    
    async def init(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the STT component if not already initialized.
        
        Args:
            api_key: Deepgram API key (optional)
        """
        if self.initialized:
            return
            
        try:
            # Create a new Deepgram streaming client optimized for telephony
            self.speech_recognizer = DeepgramStreamingSTT(
                api_key=api_key,
                language=self.language,
                sample_rate=16000,
                encoding="linear16",
                channels=1,
                interim_results=True
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with Deepgram API and language: {self.language}")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
        """
        Update ambient noise level based on audio energy.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Calculate energy of the audio
        energy = np.mean(np.abs(audio_data))
        
        # If audio is silence (very low energy), use it to update noise floor
        if energy < 0.02:  # Very quiet audio
            self.noise_samples.append(energy)
            # Keep only recent samples
            if len(self.noise_samples) > self.max_samples:
                self.noise_samples.pop(0)
            
            # Update ambient noise level (with safety floor)
            if self.noise_samples:
                # Use 95th percentile to avoid outliers
                self.ambient_noise_level = max(
                    0.005,  # Minimum threshold
                    np.percentile(self.noise_samples, 95) * 2.0  # Set threshold just above noise
                )
                logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data to reduce noise.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Preprocessed audio data
        """
        try:
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            b, a = signal.butter(4, 80/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Simple noise gate (suppress very low amplitudes)
            noise_gate_threshold = max(0.015, self.ambient_noise_level)
            noise_gate = np.where(np.abs(filtered_audio) < noise_gate_threshold, 0, filtered_audio)
            
            # 3. Normalize audio to have consistent volume
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
                
            # Log stats about the audio
            orig_energy = np.mean(np.abs(audio_data))
            proc_energy = np.mean(np.abs(normalized))
            logger.debug(f"Audio preprocessing: original energy={orig_energy:.4f}, processed energy={proc_energy:.4f}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio_data  # Return original audio if preprocessing fails
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Enhanced cleanup of transcription text by removing non-speech annotations and filler words.
        
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
    
    def is_valid_transcription(self, text: str, min_words: int = 2) -> bool:
        """
        Check if a transcription is valid and worth processing.
        
        Args:
            text: Transcription text
            min_words: Minimum word count to be considered valid
            
        Returns:
            True if the transcription is valid
        """
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if it's empty after cleaning
        if not cleaned_text:
            logger.info("Transcription contains only non-speech annotations")
            return False
        
        # Estimate confidence based on presence of uncertainty markers
        confidence_estimate = 1.0
        if "?" in text or "[" in text or "(" in text or "<" in text:
            confidence_estimate = 0.6  # Lower confidence if it contains uncertainty markers
            logger.info(f"Reduced confidence due to uncertainty markers: {text}")
            
        if confidence_estimate < 0.7:
            logger.info(f"Transcription confidence too low: {confidence_estimate}")
            return False
            
        # Check word count
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
        Transcribe an audio file with improved noise handling.
        
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
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
            audio_duration = len(audio) / sample_rate
            
            # Update ambient noise level
            self._update_ambient_noise_level(audio)
            
            # Apply audio preprocessing for noise reduction
            audio = self._preprocess_audio(audio)
            
            # Convert to bytes
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
            chunk_size = 4096  # Approximately 128ms of audio at 16kHz
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                await self.speech_recognizer.process_audio_chunk(chunk, store_result)
            
            # Stop streaming to get final results
            await self.speech_recognizer.stop_streaming()
            
            # Combine results if we have any
            if final_results:
                # Use the best final result (with highest confidence)
                best_result = max(final_results, key=lambda r: r.confidence)
                
                # Clean up the transcription
                cleaned_text = self.cleanup_transcription(best_result.text)
                
                return {
                    "transcription": cleaned_text,
                    "original_transcription": best_result.text,
                    "confidence": best_result.confidence,
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
        Transcribe audio data with improved noise handling.
        
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
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Assume 16-bit PCM format
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, list):
                audio = np.array(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Update ambient noise level
            self._update_ambient_noise_level(audio)
            
            # Apply audio preprocessing for noise reduction
            audio = self._preprocess_audio(audio)
            
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
            chunk_size = 4096  # Approximately 128ms of audio at 16kHz
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                await self.speech_recognizer.process_audio_chunk(chunk, store_result)
            
            # Stop streaming to get final results
            await self.speech_recognizer.stop_streaming()
            
            # Combine results if we have any
            if final_results:
                # Use the best final result (with highest confidence)
                best_result = max(final_results, key=lambda r: r.confidence)
                
                # Clean up the transcription
                cleaned_text = self.cleanup_transcription(best_result.text)
                
                return {
                    "transcription": cleaned_text,
                    "original_transcription": best_result.text,
                    "confidence": best_result.confidence,
                    "duration": len(audio) / 16000,  # Assuming 16kHz
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": self.is_valid_transcription(cleaned_text)
                }
            else:
                logger.warning("No transcription results obtained")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": len(audio) / 16000,  # Assuming 16kHz
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
        Process a chunk of streaming audio with improved noise handling.
        
        Args:
            audio_chunk: Audio chunk data
            callback: Optional callback for results
            
        Returns:
            Transcription result or None for interim results
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return None
        
        # Convert to numpy array if needed
        if isinstance(audio_chunk, bytes):
            # Assume 16-bit PCM format
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_chunk, list):
            audio_data = np.array(audio_chunk, dtype=np.float32)
        else:
            audio_data = audio_chunk
        
        # Update ambient noise level
        self._update_ambient_noise_level(audio_data)
        
        # Apply audio preprocessing for noise reduction
        audio_data = self._preprocess_audio(audio_data)
        
        # Convert to bytes (16-bit PCM)
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        
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
                if result.text and self.is_valid_transcription(result.text) and callback:
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
        await self.speech_recognizer.stop_streaming()
        
        # Check if we have a last result
        if hasattr(self.speech_recognizer, 'last_result') and self.speech_recognizer.last_result:
            result = self.speech_recognizer.last_result
            
            # Clean up the transcription
            cleaned_text = self.cleanup_transcription(result.text)
            
            # Log what was changed if significant
            if result.text != cleaned_text:
                logger.info(f"Cleaned final transcription: '{result.text}' -> '{cleaned_text}'")
            
            # Get the duration from the result if available
            duration = (result.end_time - result.start_time) if result.end_time > 0 else 0
            
            return cleaned_text, duration
        else:
            return "", 0.0
    
    async def process_realtime_audio_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None,
        silence_frames_threshold: int = 30
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process real-time audio stream and detect utterances with improved noise handling.
        
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
        
        # Track state
        is_speaking = False
        silence_frames = 0
        max_silence_frames = silence_frames_threshold
        
        try:
            async for audio_chunk in audio_stream:
                # Update ambient noise level
                self._update_ambient_noise_level(audio_chunk)
                
                # Apply audio preprocessing for noise reduction
                processed_chunk = self._preprocess_audio(audio_chunk)
                
                # Convert to bytes (16-bit PCM)
                audio_bytes = (processed_chunk * 32767).astype(np.int16).tobytes()
                
                # Define a result processing callback
                results = []
                
                async def process_result(result: StreamingTranscriptionResult):
                    # Store the result
                    results.append(result)
                    
                    # Call the original callback if provided
                    if callback:
                        await callback(result)
                
                # Process the audio chunk
                await self.speech_recognizer.process_audio_chunk(
                    audio_chunk=audio_bytes,
                    callback=process_result
                )
                
                # Check for speech activity
                if not is_speaking:
                    # Detect start of speech from any results
                    has_speech = any(result.text.strip() for result in results)
                    
                    if has_speech:
                        is_speaking = True
                        silence_frames = 0
                        logger.info("Speech detected, beginning transcription")
                else:
                    # Check for end of utterance (silence after speech)
                    has_speech = any(result.text.strip() for result in results)
                    
                    if not has_speech:
                        silence_frames += 1
                    else:
                        silence_frames = 0
                
                # If we've detected enough silence after speech, process the utterance
                if is_speaking and silence_frames >= max_silence_frames:
                    is_speaking = False
                    
                    # Get final transcription
                    final_text, duration = await self.end_streaming()
                    
                    # Clean up the transcription
                    cleaned_text = self.cleanup_transcription(final_text)
                    
                    # Only yield if it's a valid transcription after cleaning
                    if cleaned_text and self.is_valid_transcription(cleaned_text):
                        logger.info(f"Utterance detected: {cleaned_text}")
                        
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