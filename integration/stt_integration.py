"""
Speech-to-Text integration module for Voice AI Agent.

This module provides classes and functions for integrating speech-to-text
capabilities with the Voice AI Agent system.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator

import numpy as np

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult
from speech_to_text.utils.audio_utils import load_audio_file

logger = logging.getLogger(__name__)

class STTIntegration:
    """
    Speech-to-Text integration for Voice AI Agent.
    
    Provides an abstraction layer for speech recognition functionality,
    handling audio processing and transcription.
    """
    
    def __init__(
        self,
        speech_recognizer: StreamingWhisperASR,
        language: str = "en"
    ):
        """
        Initialize the STT integration.
        
        Args:
            speech_recognizer: Initialized StreamingWhisperASR instance
            language: Language code for speech recognition
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
    
    async def init(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the STT component if not already initialized.
        
        Args:
            model_path: Path to Whisper model (optional)
        """
        if self.initialized:
            return
            
        try:
            if not model_path:
                model_path = "tiny.en"
                
            self.speech_recognizer = StreamingWhisperASR(
                model_path=model_path,
                language=self.language,
                n_threads=4,
                chunk_size_ms=2000,
                vad_enabled=True,
                single_segment=True,
                temperature=0.0
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with model: {model_path}")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    async def transcribe_audio_file(
        self,
        audio_file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
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
            
            # Process audio based on duration
            is_short_audio = audio_duration < 5.0  # Less than 5 seconds
            
            if is_short_audio:
                logger.info(f"Processing short audio file: {audio_duration:.2f}s")
                result = await self._transcribe_short_audio(audio, callback)
            else:
                logger.info(f"Processing normal-length audio file: {audio_duration:.2f}s")
                result = await self._transcribe_normal_audio(audio, sample_rate, callback)
            
            # Add timing information
            result["processing_time"] = time.time() - start_time
            result["audio_duration"] = audio_duration
            
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
        is_short_audio: bool = False,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio data.
        
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
                # Convert bytes to float array (implementation depends on your audio format)
                audio = np.frombuffer(audio_data, dtype=np.float32)
            elif isinstance(audio_data, list):
                audio = np.array(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Auto-detect short audio if not specified
            if not is_short_audio and len(audio) < 5 * 16000:  # Less than 5 seconds at 16kHz
                is_short_audio = True
                logger.debug(f"Auto-detected short audio: {len(audio)/16000:.2f}s")
            
            # Process based on audio length
            if is_short_audio:
                result = await self._transcribe_short_audio(audio, callback)
            else:
                result = await self._transcribe_audio_chunk(audio, callback)
            
            # Add timing information
            result["processing_time"] = time.time() - start_time
            
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
        
        self.speech_recognizer.start_streaming()
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
        
        # Convert to numpy array if needed
        if isinstance(audio_chunk, bytes):
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
        elif isinstance(audio_chunk, list):
            audio_data = np.array(audio_chunk, dtype=np.float32)
        else:
            audio_data = audio_chunk
        
        # Process the audio chunk
        return await self.speech_recognizer.process_audio_chunk(
            audio_chunk=audio_data,
            callback=callback
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
        
        return await self.speech_recognizer.stop_streaming()
    
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
        self.speech_recognizer.start_streaming()
        
        # Track state
        is_speaking = False
        silence_frames = 0
        max_silence_frames = silence_frames_threshold
        
        try:
            async for audio_chunk in audio_stream:
                # Process the audio chunk
                result = await self.speech_recognizer.process_audio_chunk(
                    audio_chunk=audio_chunk,
                    callback=callback
                )
                
                # Check for speech activity
                if not is_speaking:
                    # Detect start of speech
                    if result and result.text.strip():
                        is_speaking = True
                        silence_frames = 0
                        logger.info("Speech detected, beginning transcription")
                else:
                    # Check for end of utterance (silence after speech)
                    if not result or not result.text.strip():
                        silence_frames += 1
                    else:
                        silence_frames = 0
                
                # If we've detected enough silence after speech, process the utterance
                if is_speaking and silence_frames >= max_silence_frames:
                    is_speaking = False
                    
                    # Get final transcription
                    transcription, duration = await self.speech_recognizer.stop_streaming()
                    
                    if transcription.strip():
                        logger.info(f"Utterance detected: {transcription}")
                        
                        # Yield the result
                        yield {
                            "transcription": transcription,
                            "duration": duration,
                            "is_final": True
                        }
                    
                    # Reset for next utterance
                    self.speech_recognizer.start_streaming()
                    silence_frames = 0
        
        except Exception as e:
            logger.error(f"Error in real-time audio processing: {e}")
            yield {"error": str(e)}
        
        finally:
            # Clean up
            await self.speech_recognizer.stop_streaming()
    
    async def _transcribe_short_audio(
        self,
        audio: np.ndarray,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Transcribe short audio with optimized parameters.
        
        Args:
            audio: Audio data
            callback: Optional callback for interim results
            
        Returns:
            Dictionary with transcription results
        """
        # Save original settings
        original_vad = self.speech_recognizer.vad_enabled
        
        # Use simple approach for short audio
        self.speech_recognizer.vad_enabled = False  # Disable VAD for short audio
        
        # Try multiple approaches to get transcription
        transcription = ""
        duration = 0
        
        # First attempt
        try:
            self.speech_recognizer.start_streaming()
            await self.speech_recognizer.process_audio_chunk(audio, callback)
            transcription, duration = await self.speech_recognizer.stop_streaming()
        except Exception as e:
            logger.warning(f"First transcription attempt failed: {e}")
        
        # If first attempt failed, try again with higher temperature
        if not transcription or transcription.strip() == "":
            try:
                logger.info("First attempt yielded no transcription, trying again")
                self.speech_recognizer.start_streaming()
                await self.speech_recognizer.process_audio_chunk(audio, callback)
                transcription, duration = await self.speech_recognizer.stop_streaming()
            except Exception as e:
                logger.warning(f"Second transcription attempt failed: {e}")
        
        # Restore original settings
        self.speech_recognizer.vad_enabled = original_vad
        
        # Return result
        return {
            "transcription": transcription,
            "duration": duration,
            "is_final": True
        }
    
    async def _transcribe_normal_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        callback: Optional[Callable] = None,
        chunk_size_ms: int = 1000,
        simulate_realtime: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe normal-length audio using chunking.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            callback: Optional callback for interim results
            chunk_size_ms: Size of chunks in milliseconds
            simulate_realtime: Whether to simulate real-time processing
            
        Returns:
            Dictionary with transcription results
        """
        # Calculate chunk size in samples
        chunk_size = int(sample_rate * chunk_size_ms / 1000)
        
        # Split audio into chunks
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        
        # Storage for transcriptions
        transcriptions = []
        
        # Process each chunk
        async def transcription_callback(result: StreamingTranscriptionResult):
            if result.text.strip():
                transcriptions.append(result.text)
                logger.info(f"Interim transcription: {result.text}")
                
                # Call user callback if provided
                if callback:
                    await callback(result)
        
        # Start streaming
        self.speech_recognizer.start_streaming()
        
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(audio))
            chunk = audio[chunk_start:chunk_end]
            
            # Process chunk
            await self.speech_recognizer.process_audio_chunk(
                audio_chunk=chunk,
                callback=transcription_callback
            )
            
            # Simulate real-time processing if requested
            if simulate_realtime and i < num_chunks - 1:
                await asyncio.sleep(chunk_size_ms / 1000)
        
        # Get final transcription
        final_text, duration = await self.speech_recognizer.stop_streaming()
        
        # Return result
        return {
            "transcription": final_text,
            "interim_transcriptions": transcriptions,
            "duration": duration,
            "is_final": True,
            "num_chunks": num_chunks
        }
    
    async def _transcribe_audio_chunk(
        self,
        audio: np.ndarray,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio: Audio data
            callback: Optional callback for interim results
            
        Returns:
            Dictionary with transcription results
        """
        # Start streaming
        self.speech_recognizer.start_streaming()
        
        # Process audio
        await self.speech_recognizer.process_audio_chunk(audio, callback)
        
        # Get final transcription
        final_text, duration = await self.speech_recognizer.stop_streaming()
        
        # Return result
        return {
            "transcription": final_text,
            "duration": duration,
            "is_final": True
        }