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

# Assuming AudioChunker and ChunkMetadata are defined elsewhere or imported
# For demonstration, let's add simple placeholders if they are not available
try:
    from speech_to_text.streaming.chunker import AudioChunker, ChunkMetadata
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("AudioChunker and ChunkMetadata not found. Using dummy implementations.")

    @dataclass
    class ChunkMetadata:
        chunk_id: int
        start_sample: int
        end_sample: int
        sample_rate: int
        is_first_chunk: bool

        @property
        def start_time(self) -> float:
            return self.start_sample / self.sample_rate

        @property
        def end_time(self) -> float:
            return self.end_sample / self.sample_rate

    class AudioChunker:
        def __init__(self, sample_rate, chunk_size_ms, overlap_ms, silence_threshold, min_silence_ms, max_chunk_size_ms):
            self.sample_rate = sample_rate
            self.chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
            self.overlap_samples = int(sample_rate * overlap_ms / 1000)
            self.buffer = np.array([], dtype=np.float32)
            self.processed_samples = 0
            # Simplified logic for demonstration
            self.chunk_queue = []

        def add_audio(self, audio_chunk: np.ndarray) -> bool:
            self.buffer = np.concatenate([self.buffer, audio_chunk])
            # Simplified: always create a chunk if buffer is large enough
            if len(self.buffer) >= self.chunk_size_samples:
                chunk = self.buffer[:self.chunk_size_samples]
                self.chunk_queue.append(chunk)
                # Simulate overlap
                self.buffer = self.buffer[self.chunk_size_samples - self.overlap_samples:]
                return True
            return False

        def get_chunk(self) -> Optional[np.ndarray]:
            if self.chunk_queue:
                chunk = self.chunk_queue.pop(0)
                self.processed_samples += len(chunk) # Simplified update
                return chunk
            return None

        def get_final_chunk(self) -> Optional[np.ndarray]:
            if len(self.buffer) > 0:
                chunk = self.buffer
                self.processed_samples += len(chunk) # Simplified update
                self.buffer = np.array([], dtype=np.float32)
                return chunk
            return None

        def reset(self):
            self.buffer = np.array([], dtype=np.float32)
            self.processed_samples = 0
            self.chunk_queue = []


from pywhispercpp.model import Model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config for logging


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
        self.initial_prompt = initial_prompt # Note: initial_prompt is not directly used in pywhispercpp's transcribe method
        self.max_tokens = max_tokens # Note: max_tokens might not be settable directly in pywhispercpp like this
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
            self.model = Model(model_path, n_threads=n_threads) # Pass n_threads at init

            # Set language if provided
            if language:
                try:
                    self.model.language = language
                    logger.info(f"Set model language to: {language}")
                except Exception as e:
                    logger.warning(f"Could not set language to {language}: {e}")

            # Store transcription parameters (some might be set directly on model)
            self.transcribe_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens, # Keep track, but might not be directly usable
                "initial_prompt": self.initial_prompt # Keep track, but might not be directly usable
            }

            # Track which parameters we can safely set directly on the model
            self.can_set_single_segment = hasattr(self.model, 'single_segment')
            self.can_set_no_context = hasattr(self.model, 'no_context')
            self.can_set_temperature = hasattr(self.model, 'temperature')

            # Try to set the parameters directly on the model instance if possible
            if self.can_set_temperature:
                try:
                    self.model.temperature = self.temperature
                    logger.info(f"Set model temperature to: {self.temperature}")
                except Exception as e:
                    logger.warning(f"Could not set temperature to {self.temperature}: {e}")
                    self.can_set_temperature = False # Mark as not settable if error occurs

            if self.can_set_single_segment:
                try:
                    self.model.single_segment = self.single_segment
                    logger.info(f"Set model single_segment to: {self.single_segment}")
                except Exception as e:
                    logger.warning(f"Could not set single_segment to {self.single_segment}: {e}")
                    self.can_set_single_segment = False

            if self.can_set_no_context:
                try:
                    self.model.no_context = self.no_context
                    logger.info(f"Set model no_context to: {self.no_context}")
                except Exception as e:
                    logger.warning(f"Could not set no_context to {self.no_context}: {e}")
                    self.can_set_no_context = False

            # Log the effective parameters
            param_str = ", ".join([f"{k}={v}" for k, v in self.transcribe_params.items()])
            logger.info(f"Tracking transcription parameters: {param_str}")
            logger.info(f"Model supports direct setting of: temp={self.can_set_temperature}, single_seg={self.can_set_single_segment}, no_ctx={self.can_set_no_context}")

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            logger.info("Falling back to base.en model")
            # Fallback initialization
            self.model = Model("base.en", n_threads=n_threads)
            self.transcribe_params = {} # Reset params on fallback
            # Re-check capabilities for fallback model
            self.can_set_single_segment = hasattr(self.model, 'single_segment')
            self.can_set_no_context = hasattr(self.model, 'no_context')
            self.can_set_temperature = hasattr(self.model, 'temperature')

        # Set translation if needed
        if translate:
            if hasattr(self.model, 'set_translate'):
                 try:
                    self.model.set_translate(True)
                    logger.info("Translation enabled")
                 except Exception as e:
                    logger.warning(f"Could not enable translation: {e}")
            else:
                logger.warning("Model object does not support set_translate method.")


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
            audio_chunk: Audio data as numpy array (expects float32)
            callback: Optional async callback function for results

        Returns:
            StreamingTranscriptionResult or None if no result available
        """
        # Start streaming if not already started
        if not self.is_streaming:
            self.start_streaming()

        # Ensure audio chunk is float32
        if audio_chunk.dtype != np.float32:
            # Try converting common types
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype == np.int32:
                audio_chunk = audio_chunk.astype(np.float32) / 2147483648.0
            elif audio_chunk.dtype == np.uint8: # Assuming unsigned 8-bit PCM
                 audio_chunk = (audio_chunk.astype(np.float32) - 128.0) / 128.0
            else:
                 # Fallback conversion, might not be correct range
                 logger.warning(f"Unsupported audio dtype {audio_chunk.dtype}, attempting direct conversion to float32.")
                 audio_chunk = audio_chunk.astype(np.float32)


        # Ensure audio is in range [-1.0, 1.0] - normalization might be needed
        # Be cautious with blanket normalization; it can amplify noise.
        # max_val = np.max(np.abs(audio_chunk))
        # if max_val > 1.0:
        #     logger.warning(f"Audio chunk max abs value {max_val} > 1.0. Normalizing.")
        #     audio_chunk = audio_chunk / max_val
        # Clipping might be safer than normalization for preserving dynamics:
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)


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
            chunk: Audio chunk as numpy array (should be float32)
            callback: Optional async callback for results

        Returns:
            StreamingTranscriptionResult or None
        """
        # Generate chunk metadata
        self.last_chunk_id += 1
        chunk_id = self.last_chunk_id

        # Note: chunker.processed_samples needs careful management based on overlap
        # Using simple time tracking might be more robust for start/end times
        current_stream_time = time.time() - self.streaming_start_time
        chunk_duration = len(chunk) / self.sample_rate

        # Rough estimate of start/end times based on stream progress
        # This might drift if processing falls behind real-time.
        # A more accurate approach would involve tracking absolute sample counts.
        chunk_start_time = current_stream_time # Approximate start relative to stream start
        chunk_end_time = chunk_start_time + chunk_duration

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            # These sample counts might not be accurate with the dummy chunker
            start_sample=int(chunk_start_time * self.sample_rate),
            end_sample=int(chunk_end_time * self.sample_rate),
            sample_rate=self.sample_rate,
            is_first_chunk=(chunk_id == 1),
        )

        # Perform voice activity detection if enabled
        contains_speech = True
        if self.vad_enabled:
            contains_speech = self._detect_speech(chunk, threshold=self.chunker.silence_threshold) # Use chunker's threshold

        # If no speech detected, skip transcription
        if not contains_speech:
            logger.debug(f"No speech detected in chunk {chunk_id}, skipping transcription")
            result = StreamingTranscriptionResult(
                text="",
                is_final=True,
                confidence=0.0,
                start_time=chunk_start_time, # Use calculated time
                end_time=chunk_end_time,     # Use calculated time
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

            # Use the safe transcription function
            transcribe_func = lambda: self._safe_transcribe(chunk)
            segments = await loop.run_in_executor(self.executor, transcribe_func)

            processing_time = time.time() - start_time

            logger.debug(f"Processed chunk {chunk_id} in {processing_time:.3f}s")

            # Handle transcription results
            if not segments:
                logger.debug(f"No segments returned for chunk {chunk_id}")
                return None

            # Combine results from all segments
            # Filter out potential placeholder text from whisper.cpp
            meaningful_segments = [
                segment for segment in segments
                if segment.text.strip() and segment.text.strip() not in ["[BLANK_AUDIO]", "(silent)", "[(silent)]"]
            ]

            if not meaningful_segments:
                 logger.debug(f"No meaningful text in segments for chunk {chunk_id}")
                 return None

            combined_text = " ".join(segment.text.strip() for segment in meaningful_segments)


            # Estimate confidence (pywhispercpp doesn't provide it directly)
            # We could use segment probability if available, or just use 1.0
            avg_confidence = 1.0 # Placeholder

            # Create streaming result
            result = StreamingTranscriptionResult(
                text=combined_text,
                is_final=True,  # For now, all results are final per chunk
                confidence=avg_confidence,
                start_time=chunk_start_time, # Use calculated time
                end_time=chunk_end_time,     # Use calculated time
                chunk_id=chunk_id,
            )

            # Update partial text for the whole stream
            self.partial_text += (" " + combined_text) if self.partial_text else combined_text

            # Call the callback if provided
            if callback and combined_text:
                await callback(result)

            return result

        except Exception as e:
            logger.error(f"Error in transcription for chunk {chunk_id}: {e}", exc_info=True)
            processing_time = time.time() - start_time
            logger.debug(f"Failed chunk {chunk_id} after {processing_time:.3f}s")
            return None

    # --------------------------------------------------
    # --- REPLACED FUNCTION START ---
    # --------------------------------------------------
    def _safe_transcribe(self, audio_data):
        """
        Safely transcribe audio data, handling parameter compatibility issues.
        """
        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Handle short audio
        # min_audio_length = self.sample_rate * 1.0  # 1 second - Adjust as needed, 0.1s might be enough
        min_audio_length_samples = int(self.sample_rate * 0.1) # Whisper needs at least ~30ms, 100ms is safer

        if len(audio_data) < min_audio_length_samples:
            # Pad with silence if too short
            required_padding = min_audio_length_samples - len(audio_data)
            logger.info(f"Audio too short ({len(audio_data)/self.sample_rate:.3f}s), padding with {required_padding/self.sample_rate:.3f}s silence")
            padding = np.zeros(required_padding, dtype=np.float32)
            audio_data = np.concatenate([audio_data, padding])

        # Set parameters safely only if they are known to be settable
        if self.can_set_temperature:
            try:
                self.model.temperature = self.temperature
            except Exception as e:
                 logger.warning(f"Failed to set temperature dynamically: {e}")
                 self.can_set_temperature = False # Disable future attempts if it fails once

        if self.can_set_single_segment:
            try:
                self.model.single_segment = self.single_segment
            except Exception as e:
                 logger.warning(f"Failed to set single_segment dynamically: {e}")
                 self.can_set_single_segment = False

        if self.can_set_no_context:
            try:
                self.model.no_context = self.no_context
            except Exception as e:
                 logger.warning(f"Failed to set no_context dynamically: {e}")
                 self.can_set_no_context = False

        # Transcribe safely
        try:
            # Note: pywhispercpp's transcribe doesn't typically accept many args directly
            # Pass only audio_data. Other params are usually set on the model object beforehand.
            segments = self.model.transcribe(audio_data)
            return segments
        except TypeError as e:
            # This might catch errors if pywhispercpp API changes or expects different args
            logger.warning(f"Transcription TypeError: {e}. Retrying basic transcribe.")
            try:
                # Retry with minimal call
                segments = self.model.transcribe(audio_data)
                return segments
            except Exception as e2:
                logger.error(f"Second transcription attempt failed: {e2}", exc_info=True)
                return []
        except Exception as e:
             logger.error(f"General transcription error: {e}", exc_info=True)
             return []
    # --------------------------------------------------
    # --- REPLACED FUNCTION END ---
    # --------------------------------------------------

    def _detect_speech(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Simple voice activity detection based on root mean square energy.

        Args:
            audio: Audio data (float32 numpy array)
            threshold: Energy threshold for speech detection

        Returns:
            True if speech energy is above threshold, False otherwise
        """
        if audio.size == 0:
            return False
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio**2))
        # logger.debug(f"Chunk RMS energy: {rms:.4f}, Threshold: {threshold:.4f}")
        return rms > threshold

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
        Stop the current streaming session and process any remaining audio.

        Returns:
            Tuple of (final_text, stream_duration)
        """
        if not self.is_streaming:
            logger.warning("stop_streaming called but not currently streaming.")
            return self.partial_text.strip(), self.stream_duration # Return current state

        logger.info("Stopping streaming session...")
        # Process any remaining audio in the buffer
        final_chunk = self.chunker.get_final_chunk()
        final_text = self.partial_text # Start with text accumulated so far

        if final_chunk is not None and len(final_chunk) > 0:
            logger.info(f"Processing final audio chunk of length {len(final_chunk)} samples...")
            # Use _process_chunk which handles VAD and transcription
            result = await self._process_chunk(final_chunk) # Don't need callback here
            # Note: _process_chunk already updates self.partial_text if successful
            # We retrieve the potentially updated self.partial_text below
            final_text = self.partial_text # Get potentially updated text

        self.stream_duration = time.time() - self.streaming_start_time
        self.is_streaming = False

        # Clean up final text
        cleaned_final_text = final_text.strip()
        # Remove placeholder text if it's the only thing transcribed
        placeholders = ["[BLANK_AUDIO]", "(silent)", "[(silent)]"]
        if cleaned_final_text in placeholders:
            cleaned_final_text = ""

        logger.info(f"Stopped streaming session after {self.stream_duration:.2f}s. Final text length: {len(cleaned_final_text)}")
        return cleaned_final_text, self.stream_duration

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
            initial_prompt: New initial prompt (Note: Affects future logic if used, not directly passed to pywhispercpp transcribe)
            max_tokens: New max tokens value (Note: Affects future logic if used, not directly passed to pywhispercpp transcribe)
            no_context: New no_context value
            single_segment: New single_segment value
        """
        updated_params = {}
        if temperature is not None:
            self.temperature = temperature
            self.transcribe_params["temperature"] = temperature
            if self.can_set_temperature:
                try:
                    self.model.temperature = temperature
                    updated_params['temperature'] = temperature
                except Exception as e:
                    logger.warning(f"Could not update temperature: {e}")
                    self.can_set_temperature = False # Assume it can't be set if error

        if initial_prompt is not None:
            self.initial_prompt = initial_prompt
            self.transcribe_params["initial_prompt"] = initial_prompt
            updated_params['initial_prompt'] = initial_prompt # Track even if not directly used by model

        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.transcribe_params["max_tokens"] = max_tokens
            updated_params['max_tokens'] = max_tokens # Track even if not directly used by model

        if no_context is not None:
            self.no_context = no_context
            # self.transcribe_params["no_context"] = no_context # Not needed in transcribe_params
            if self.can_set_no_context:
                try:
                    self.model.no_context = no_context
                    updated_params['no_context'] = no_context
                except Exception as e:
                    logger.warning(f"Could not update no_context: {e}")
                    self.can_set_no_context = False

        if single_segment is not None:
            self.single_segment = single_segment
            # self.transcribe_params["single_segment"] = single_segment # Not needed in transcribe_params
            if self.can_set_single_segment:
                try:
                    self.model.single_segment = single_segment
                    updated_params['single_segment'] = single_segment
                except Exception as e:
                     logger.warning(f"Could not update single_segment: {e}")
                     self.can_set_single_segment = False

        if updated_params:
            logger.info(f"Updated transcription parameters: {updated_params}")
        else:
            logger.info("No parameters were updated (or supported for dynamic update).")


    def set_parameter_preset(self, preset: str):
        """
        Set parameters according to a predefined preset.

        Args:
            preset: Name of the preset to use
        """
        if preset not in PARAMETER_PRESETS:
            logger.warning(f"Preset '{preset}' not found. Available presets: {list(PARAMETER_PRESETS.keys())}")
            return

        logger.info(f"Applying parameter preset: {preset}")
        preset_params = PARAMETER_PRESETS[preset]
        self.update_parameters(
            temperature=preset_params["temperature"],
            initial_prompt=preset_params["initial_prompt"],
            max_tokens=preset_params["max_tokens"],
            no_context=preset_params["no_context"],
            single_segment=preset_params["single_segment"]
        )


    def __del__(self):
        """Clean up resources."""
        logger.debug("Shutting down ThreadPoolExecutor in StreamingWhisperASR.")
        if hasattr(self, 'executor'):
            # Shutdown gracefully, wait for running tasks by default
            # Use wait=False for faster exit if needed, but might interrupt tasks
            self.executor.shutdown(wait=True)
        # pywhispercpp model doesn't seem to have an explicit close/del method needed

# Example usage (requires a model file and audio source)
async def example_stream():
    # Need a GGML model file (e.g., download base.en model)
    MODEL_PATH = "path/to/your/ggml-base.en.bin" # <--- CHANGE THIS

    try:
        asr = StreamingWhisperASR(model_path=MODEL_PATH, language="en", preset="default")
    except Exception as e:
         logger.error(f"Failed to initialize ASR: {e}. Make sure model path is correct.")
         return

    async def result_callback(result: StreamingTranscriptionResult):
        if result.text.strip(): # Only print if there's actual text
            print(f"[{result.start_time:.2f}-{result.end_time:.2f}] (Chunk {result.chunk_id}, Final={result.is_final}): {result.text}")

    # Simulate receiving audio chunks (e.g., from microphone or file)
    # Generate dummy audio data for demonstration
    sample_rate = 16000
    chunk_duration_s = 0.5 # Process 500ms chunks
    chunk_samples = int(sample_rate * chunk_duration_s)

    print("Simulating audio stream...")
    asr.start_streaming() # Explicitly start

    total_duration_s = 5
    num_chunks = int(total_duration_s / chunk_duration_s)

    try:
        for i in range(num_chunks):
            # Generate a sine wave chunk for demo purposes
            freq = 440.0 # A4 note
            t = np.linspace(i * chunk_duration_s, (i+1) * chunk_duration_s, chunk_samples, endpoint=False)
            audio_chunk = 0.5 * np.sin(2 * np.pi * freq * t)
            audio_chunk = audio_chunk.astype(np.float32)

            # Add some silence occasionally
            if i % 4 == 0:
                 audio_chunk[:] = 0.0 # Make this chunk silent

            # Simulate processing delay
            await asyncio.sleep(chunk_duration_s * 0.8) # Simulate near real-time arrival

            await asr.process_audio_chunk(audio_chunk, callback=result_callback)

        # Stop streaming and get final result
        final_transcript, duration = await asr.stop_streaming()
        print("\n--- End of Stream ---")
        print(f"Total stream duration: {duration:.2f}s")
        print(f"Final Transcript: {final_transcript}")

    except Exception as e:
        logger.error(f"An error occurred during streaming: {e}", exc_info=True)
        # Ensure stop is called even on error to clean up
        await asr.stop_streaming()


if __name__ == "__main__":
    print("Running StreamingWhisperASR example...")
    # Make sure you have a model file downloaded and update MODEL_PATH above
    # Example: wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
    asyncio.run(example_stream())