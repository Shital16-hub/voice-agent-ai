
"""
Streaming wrapper for Whisper.cpp using pywhispercpp.
"""
import time
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Awaitable, Any, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

from pywhispercpp.model import Model
from speech_to_text.streaming.chunker import AudioChunker, ChunkMetadata

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float
    start_time: float
    end_time: float
    chunk_id: int

# Define parameter presets for experimentation
PARAMETER_PRESETS = {
    "default": {
        "temperature": 0.2,
        "initial_prompt": "",
        "max_tokens": 100,
        "no_context": False,
        "single_segment": True
    },
    "creative": {
        "temperature": 0.6,
        "initial_prompt": "Creative interpretation of the audio:",
        "max_tokens": 0,
        "no_context": False,
        "single_segment": True
    },
    "structured": {
        "temperature": 0.0,
        "initial_prompt": "Transcript formatted as a dialogue:",
        "max_tokens": 100,
        "no_context": False,
        "single_segment": True
    },
    "technical": {
        "temperature": 0.2,
        "initial_prompt": "Technical discussion transcript:",
        "max_tokens": 0,
        "no_context": False,
        "single_segment": True
    },
    "meeting": {
        "temperature": 0.3,
        "initial_prompt": "Minutes from a business meeting:",
        "max_tokens": 150,
        "no_context": False,
        "single_segment": True
    }
}

class StreamingWhisperASR:
    """
    Streaming speech recognition using Whisper.cpp via pywhispercpp.
    
    This class handles the real-time streaming of audio data,
    chunking, and recognition using the Whisper model.
    """
    
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        language: str = "en",
        n_threads: int = 4,
        chunk_size_ms: int = 1000,
        overlap_ms: int = 200,
        silence_threshold: float = 0.01,
        min_silence_ms: int = 500,
        max_chunk_size_ms: int = 30000,
        vad_enabled: bool = True,
        translate: bool = False,
        # Add the parameters for experimentation
        temperature: float = 0.0,
        initial_prompt: Optional[str] = None,
        max_tokens: int = 0,
        no_context: bool = False,
        single_segment: bool = True,
        # Add preset parameter
        preset: Optional[str] = None
    ):
        """
        Initialize StreamingWhisperASR.
        
        Args:
            model_path: Path to the Whisper model file
            sample_rate: Audio sample rate in Hz
            language: Language code for recognition
            n_threads: Number of CPU threads to use
            chunk_size_ms: Size of each audio chunk in milliseconds
            overlap_ms: Overlap between chunks in milliseconds
            silence_threshold: Threshold for silence detection
            min_silence_ms: Minimum silence duration for chunking
            max_chunk_size_ms: Maximum chunk size in milliseconds
            vad_enabled: Whether to use voice activity detection
            translate: Whether to translate non-English to English
            temperature: Controls creativity in transcription (higher = more creative)
            initial_prompt: Provides context to guide the transcription
            max_tokens: Limits the number of tokens per segment
            no_context: Controls whether to use previous transcription as context
            single_segment: Enabled for better streaming performance
            preset: Name of parameter preset to use (overrides individual parameters)
        """
        self.sample_rate = sample_rate
        self.vad_enabled = vad_enabled
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # If a preset is specified, use its parameters
        if preset and preset in PARAMETER_PRESETS:
            logger.info(f"Using parameter preset: {preset}")
            preset_params = PARAMETER_PRESETS[preset]
            temperature = preset_params["temperature"]
            initial_prompt = preset_params["initial_prompt"]
            max_tokens = preset_params["max_tokens"]
            no_context = preset_params["no_context"]
            single_segment = preset_params["single_segment"]
        
        # Store the parameters for transcription
        self.temperature = temperature
        self.initial_prompt = initial_prompt
        self.max_tokens = max_tokens
        self.no_context = no_context
        self.single_segment = single_segment
        
        # Initialize the audio chunker
        self.chunker = AudioChunker(
            sample_rate=sample_rate,
            chunk_size_ms=chunk_size_ms,
            overlap_ms=overlap_ms,
            silence_threshold=silence_threshold,
            min_silence_ms=min_silence_ms,
            max_chunk_size_ms=max_chunk_size_ms,
        )
        
        # Initialize the Whisper model using pywhispercpp
        try:
            logger.info(f"Loading model: {model_path}")
            
            # Initialize the model
            self.model = Model(model_path)
            
            # Set language if provided
            if language:
                try:
                    self.model.language = language
                except Exception as e:
                    logger.warning(f"Could not set language to {language}: {e}")
            
            # Set number of threads
            try:
                self.model.n_threads = n_threads
            except Exception as e:
                logger.warning(f"Could not set n_threads to {n_threads}: {e}")
            
            # Store transcription parameters
            self.transcribe_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # Track which parameters we can safely set directly on the model
            self.can_set_single_segment = True
            self.can_set_no_context = True
            
            # Try to set the remaining parameters - some might not be supported
            try:
                if single_segment:
                    self.model.single_segment = single_segment
            except Exception:
                logger.warning("single_segment parameter not directly supported, will handle in transcribe")
                self.can_set_single_segment = False
            
            try:
                if no_context:
                    self.model.no_context = no_context
            except Exception:
                logger.warning("no_context parameter not directly supported, will handle in transcribe")
                self.can_set_no_context = False
            
            # Log the parameters we're using
            param_str = ", ".join([f"{k}={v}" for k, v in self.transcribe_params.items() if v is not None])
            logger.info(f"Model parameters: {param_str}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            logger.info("Falling back to base.en model")
            self.model = Model("base.en")
            self.transcribe_params = {}
        
        # Set translation if needed
        if translate:
            try:
                self.model.set_translate(True)
                logger.info("Translation enabled")
            except Exception as e:
                logger.warning(f"Could not enable translation: {e}")
        
        # Tracking state
        self.is_streaming = False
        self.last_chunk_id = 0
        self.partial_text = ""
        self.streaming_start_time = 0
        self.stream_duration = 0
        
        logger.info(f"Initialized StreamingWhisperASR with model {model_path}")
    
    async def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process an audio chunk for streaming recognition.
        
        Args:
            audio_chunk: Audio data as numpy array
            callback: Optional async callback function for results
            
        Returns:
            StreamingTranscriptionResult or None if no result available
        """
        # Start streaming if not already started
        if not self.is_streaming:
            self.start_streaming()
        
        # Ensure audio chunk is float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Ensure audio is in range [-1.0, 1.0]
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val
        
        # Add the audio to the chunker
        has_chunk = self.chunker.add_audio(audio_chunk)
        
        if not has_chunk:
            return None
        
        # Get the next chunk for processing
        chunk = self.chunker.get_chunk()
        if chunk is None:
            return None
        
        # Process the chunk
        result = await self._process_chunk(chunk, callback)
        return result
    
    async def _process_chunk(
        self,
        chunk: np.ndarray,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a single audio chunk.
        
        Args:
            chunk: Audio chunk as numpy array
            callback: Optional async callback for results
            
        Returns:
            StreamingTranscriptionResult or None
        """
        # Generate chunk metadata
        self.last_chunk_id += 1
        chunk_id = self.last_chunk_id
        
        start_sample = self.chunker.processed_samples
        end_sample = start_sample + len(chunk)
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=self.sample_rate,
            is_first_chunk=(chunk_id == 1),
        )
        
        # Perform voice activity detection if enabled
        contains_speech = True
        if self.vad_enabled:
            contains_speech = self._detect_speech(chunk)
        
        # If no speech detected, skip transcription
        if not contains_speech:
            logger.debug(f"No speech detected in chunk {chunk_id}, skipping transcription")
            result = StreamingTranscriptionResult(
                text="",
                is_final=True,
                confidence=0.0,
                start_time=metadata.start_time,
                end_time=metadata.end_time,
                chunk_id=chunk_id,
            )
            
            if callback:
                await callback(result)
            
            return result
        
        # Process the chunk with Whisper
        start_time = time.time()
        
        try:
            # Run in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            # Use a safe transcription approach that works with pywhispercpp
            transcribe_func = lambda: self._safe_transcribe(chunk)
            segments = await loop.run_in_executor(self.executor, transcribe_func)
            
            processing_time = time.time() - start_time
            
            logger.debug(f"Processed chunk {chunk_id} in {processing_time:.3f}s")
            
            # Handle transcription results
            if not segments:
                return None
            
            # Combine results from all segments
            combined_text = " ".join(segment.text for segment in segments if segment.text.strip() and segment.text != "[BLANK_AUDIO]")
            
            # If we still don't have any meaningful text, return None
            if not combined_text.strip():
                return None
            
            # Create streaming result
            result = StreamingTranscriptionResult(
                text=combined_text,
                is_final=True,  # For now, all results are final
                confidence=1.0,  # pywhispercpp doesn't provide confidence values
                start_time=metadata.start_time,
                end_time=metadata.end_time,
                chunk_id=chunk_id,
            )
            
            # Update partial text
            self.partial_text += " " + combined_text if combined_text else ""
            
            # Call the callback if provided
            if callback and combined_text.strip():
                await callback(result)
            
            return result if combined_text.strip() else None
            
        except Exception as e:
            logger.error(f"Error in transcription for chunk {chunk_id}: {e}")
            processing_time = time.time() - start_time
            logger.debug(f"Failed chunk {chunk_id} after {processing_time:.3f}s")
            return None
    
    def _safe_transcribe(self, audio_data):
        """
        Safely transcribe audio data, handling parameter compatibility issues.
        
        Args:
            audio_data: Audio data to transcribe
            
        Returns:
            List of transcription segments
        """
        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure audio length is sufficient
        if len(audio_data) < self.sample_rate * 0.1:  # Less than 0.1 second
            logger.debug("Audio chunk too short for transcription")
            return []
        
        # Set parameters safely using only what's supported
        self.model.temperature = self.temperature
        
        # Some parameters might need to be set at transcription time
        # but pywhispercpp doesn't support all parameters directly
        
        # Only set parameters that the model supports, ignoring those it doesn't
        try:
            if self.can_set_single_segment and self.single_segment:
                self.model.single_segment = self.single_segment
        except:
            pass
            
        try:
            if self.can_set_no_context and self.no_context:
                self.model.no_context = self.no_context
        except:
            pass
        
        # Transcribe safely using only the audio data argument
        # This avoids the parameter compatibility issues
        try:
            segments = self.model.transcribe(audio_data)
            return segments
        except TypeError as e:
            # If we get a parameter error, try the simplest form of transcription
            logger.warning(f"Transcription parameter error: {e}, trying simplified approach")
            try:
                segments = self.model.transcribe(audio_data)
                return segments
            except Exception as e2:
                logger.error(f"Second transcription attempt failed: {e2}")
                return []
    
    def _detect_speech(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Simple voice activity detection.
        
        Args:
            audio: Audio data
            threshold: Energy threshold for speech detection
            
        Returns:
            True if speech is detected, False otherwise
        """
        # Simple energy-based VAD
        energy = np.abs(audio).mean()
        return energy > threshold
    
    def start_streaming(self):
        """Start a new streaming session."""
        self.is_streaming = True
        self.last_chunk_id = 0
        self.partial_text = ""
        self.chunker.reset()
        self.streaming_start_time = time.time()
        logger.info("Started new streaming session")
    
    async def stop_streaming(self) -> Tuple[str, float]:
        """
        Stop the current streaming session.
        
        Returns:
            Tuple of (final_text, stream_duration)
        """
        if not self.is_streaming:
            return "", 0.0
        
        # Process any remaining audio
        chunk = self.chunker.get_final_chunk()
        final_text = self.partial_text
        
        if chunk is not None and len(chunk) > 0:
            result = await self._process_chunk(chunk)
            if result and result.text not in ["[BLANK_AUDIO]", ""]:
                final_text += " " + result.text
        
        self.stream_duration = time.time() - self.streaming_start_time
        self.is_streaming = False
        
        # Return empty string if only blank audio detected
        if final_text.strip() == "[BLANK_AUDIO]":
            final_text = ""
        
        logger.info(f"Stopped streaming session after {self.stream_duration:.2f}s")
        return final_text.strip(), self.stream_duration
    
    def update_parameters(self, 
                        temperature: Optional[float] = None,
                        initial_prompt: Optional[str] = None,
                        max_tokens: Optional[int] = None,
                        no_context: Optional[bool] = None,
                        single_segment: Optional[bool] = None):
        """
        Update transcription parameters mid-session.
        
        Args:
            temperature: New temperature value
            initial_prompt: New initial prompt
            max_tokens: New max tokens value
            no_context: New no_context value
            single_segment: New single_segment value
        """
        if temperature is not None:
            self.temperature = temperature
            self.transcribe_params["temperature"] = temperature
            try:
                self.model.temperature = temperature
            except:
                pass
        
        if initial_prompt is not None:
            self.initial_prompt = initial_prompt
            self.transcribe_params["initial_prompt"] = initial_prompt
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.transcribe_params["max_tokens"] = max_tokens
        
        if no_context is not None:
            self.no_context = no_context
            self.transcribe_params["no_context"] = no_context
            if self.can_set_no_context:
                try:
                    self.model.no_context = no_context
                except:
                    pass
        
        if single_segment is not None:
            self.single_segment = single_segment
            self.transcribe_params["single_segment"] = single_segment
            if self.can_set_single_segment:
                try:
                    self.model.single_segment = single_segment
                except:
                    pass
        
        logger.info(f"Updated transcription parameters: {self.transcribe_params}")
    
    def set_parameter_preset(self, preset: str):
        """
        Set parameters according to a predefined preset.
        
        Args:
            preset: Name of the preset to use
        """
        if preset not in PARAMETER_PRESETS:
            logger.warning(f"Preset '{preset}' not found. Available presets: {list(PARAMETER_PRESETS.keys())}")
            return
        
        preset_params = PARAMETER_PRESETS[preset]
        self.update_parameters(
            temperature=preset_params["temperature"],
            initial_prompt=preset_params["initial_prompt"],
            max_tokens=preset_params["max_tokens"],
            no_context=preset_params["no_context"],
            single_segment=preset_params["single_segment"]
        )
        
        logger.info(f"Applied parameter preset: {preset}")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
