"""
Optimized streaming wrapper for Whisper.cpp using pywhispercpp with improved
noise handling capabilities.
"""
import time
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Awaitable, Any, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import re

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
            self.silence_threshold = silence_threshold

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
        "temperature": 0.0,
        "initial_prompt": "This is a clear business conversation in English. Transcribe the exact words spoken, ignoring background noise.",
        "max_tokens": 100,
        "no_context": True,
        "single_segment": True
    },
    "noisy": {
        "temperature": 0.0,
        "initial_prompt": "This is a conversation with background noise. Ignore the noise and focus only on the spoken words.",
        "max_tokens": 100,
        "no_context": True,
        "single_segment": True
    },
    "technical": {
        "temperature": 0.0,
        "initial_prompt": "This is a technical conversation. Focus on technical terms and ignore background noises.",
        "max_tokens": 150,
        "no_context": True,
        "single_segment": True
    },
    "meeting": {
        "temperature": 0.1,
        "initial_prompt": "This is a business meeting discussion. Focus on the speech and ignore background sounds.",
        "max_tokens": 150,
        "no_context": True,
        "single_segment": True
    }
}

# Non-speech patterns to filter out
NON_SPEECH_PATTERNS = [
    r'\(.*?music.*?\)',         # (music), (tense music), etc.
    r'\(.*?wind.*?\)',          # (wind), (wind blowing), etc.
    r'\(.*?engine.*?\)',        # (engine), (engine revving), etc.
    r'\(.*?noise.*?\)',         # (noise), (background noise), etc.
    r'\(.*?sound.*?\)',         # (sound), (sounds), etc.
    r'\(.*?silence.*?\)',       # (silence), etc.
    r'\[.*?silence.*?\]',       # [silence], etc.
    r'\[.*?BLANK.*?\]',         # [BLANK_AUDIO], etc.
    r'\(.*?applause.*?\)',      # (applause), etc.
    r'\(.*?laughter.*?\)',      # (laughter), etc.
    r'\(.*?footsteps.*?\)',     # (footsteps), etc.
    r'\(.*?breathing.*?\)',     # (breathing), etc.
    r'\(.*?growling.*?\)',      # (growling), etc.
    r'\(.*?coughing.*?\)',      # (coughing), etc.
    r'\[.*?noise.*?\]',         # [noise], etc.
    r'\(.*?background.*?\)',    # (background), etc.
    r'\[.*?music.*?\]',         # [music], etc.
    r'\(.*?static.*?\)',        # (static), etc.
]

class StreamingWhisperASR:
    """
    Optimized streaming speech recognition using Whisper.cpp via pywhispercpp.

    This class handles the real-time streaming of audio data,
    chunking, and recognition using the Whisper model, with enhanced
    noise handling capabilities.
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        language: str = "en",
        n_threads: int = 4,
        chunk_size_ms: int = 2000,
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
        no_context: bool = True,
        single_segment: bool = True,
        # Add preset parameter
        preset: Optional[str] = None
    ):
        """
        Initialize StreamingWhisperASR with improved noise handling parameters.

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
        else:
            # If no preset and no initial prompt, use the optimized default
            if initial_prompt is None:
                initial_prompt = PARAMETER_PRESETS["default"]["initial_prompt"]
                logger.info(f"Using default prompt: {initial_prompt}")

        # Store the parameters for transcription
        self.temperature = temperature
        self.initial_prompt = initial_prompt
        self.max_tokens = max_tokens
        self.no_context = no_context
        self.single_segment = single_segment
        
        # Compile the non-speech pattern
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))

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
            self.model = Model(model_path, n_threads=n_threads)

            # Set language if provided
            if language:
                try:
                    self.model.language = language
                    logger.info(f"Set model language to: {language}")
                except Exception as e:
                    logger.warning(f"Could not set language to {language}: {e}")

            # Store transcription parameters
            self.transcribe_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "initial_prompt": self.initial_prompt
            }

            # Track which parameters we can safely set directly on the model
            self.can_set_single_segment = hasattr(self.model, 'single_segment')
            self.can_set_no_context = hasattr(self.model, 'no_context')
            self.can_set_temperature = hasattr(self.model, 'temperature')
            self.can_set_beam_size = hasattr(self.model, 'beam_size')
            self.can_set_patience = hasattr(self.model, 'patience')
            self.can_set_suppress_blank = hasattr(self.model, 'suppress_blank')
            self.can_set_suppress_tokens = hasattr(self.model, 'suppress_tokens')
            self.can_set_entropy_threshold = hasattr(self.model, 'entropy_threshold')
            self.can_set_initial_prompt = hasattr(self.model, 'initial_prompt')

            # Try to set the parameters directly on the model instance if possible
            if self.can_set_temperature:
                try:
                    self.model.temperature = self.temperature
                    logger.info(f"Set model temperature to: {self.temperature}")
                except Exception as e:
                    logger.warning(f"Could not set temperature to {self.temperature}: {e}")
                    self.can_set_temperature = False

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
            
            # Set initial prompt if supported
            if self.can_set_initial_prompt and self.initial_prompt:
                try:
                    self.model.initial_prompt = self.initial_prompt
                    logger.info(f"Set initial prompt: {self.initial_prompt}")
                except Exception as e:
                    logger.warning(f"Could not set initial prompt: {e}")
                    self.can_set_initial_prompt = False

            # Set beam search parameters if available
            if self.can_set_beam_size:
                try:
                    self.model.beam_size = 3  # Use beam search with 3 beams
                    logger.info("Set beam size to 3")
                except Exception as e:
                    logger.warning(f"Could not set beam size: {e}")
                    self.can_set_beam_size = False
            
            if self.can_set_patience:
                try:
                    self.model.patience = 1.0  # More patience for better hypotheses
                    logger.info("Set patience to 1.0")
                except Exception as e:
                    logger.warning(f"Could not set patience: {e}")
                    self.can_set_patience = False
            
            # Set entropy threshold if supported
            if self.can_set_entropy_threshold:
                try:
                    self.model.entropy_threshold = 2.8  # Higher threshold for uncertain segments
                    logger.info("Set entropy threshold to 2.8")
                except Exception as e:
                    logger.warning(f"Could not set entropy threshold: {e}")
                    self.can_set_entropy_threshold = False
                    
            # Suppress blank detection if supported
            if self.can_set_suppress_blank:
                try:
                    self.model.suppress_blank = True
                    logger.info("Enabled suppress_blank")
                except Exception as e:
                    logger.warning(f"Could not set suppress_blank: {e}")
                    self.can_set_suppress_blank = False

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

    def update_parameters(self, temperature=None, initial_prompt=None, no_context=None, single_segment=None, beam_size=None):
        """
        Update parameters for the Whisper model.
        
        Args:
            temperature: Temperature for sampling
            initial_prompt: Initial prompt to guide transcription
            no_context: Whether to disable using context from previous segments
            single_segment: Whether to force single segment output
            beam_size: Beam size for beam search
        """
        # Update parameters on the model if available
        if hasattr(self, 'model'):
            if temperature is not None and hasattr(self.model, 'temperature'):
                try:
                    self.model.temperature = temperature
                    logger.info(f"Updated temperature to {temperature}")
                    if hasattr(self, 'temperature'):
                        self.temperature = temperature
                except Exception as e:
                    logger.warning(f"Failed to update temperature: {e}")
            
            if initial_prompt is not None and hasattr(self.model, 'initial_prompt'):
                try:
                    self.model.initial_prompt = initial_prompt
                    logger.info(f"Updated initial_prompt")
                    if hasattr(self, 'initial_prompt'):
                        self.initial_prompt = initial_prompt
                except Exception as e:
                    logger.warning(f"Failed to update initial_prompt: {e}")
            
            if no_context is not None and hasattr(self.model, 'no_context'):
                try:
                    self.model.no_context = no_context
                    logger.info(f"Updated no_context to {no_context}")
                    if hasattr(self, 'no_context'):
                        self.no_context = no_context
                except Exception as e:
                    logger.warning(f"Failed to update no_context: {e}")
            
            if single_segment is not None and hasattr(self.model, 'single_segment'):
                try:
                    self.model.single_segment = single_segment
                    logger.info(f"Updated single_segment to {single_segment}")
                    if hasattr(self, 'single_segment'):
                        self.single_segment = single_segment
                except Exception as e:
                    logger.warning(f"Failed to update single_segment: {e}")
                
            if beam_size is not None and hasattr(self.model, 'beam_size'):
                try:
                    self.model.beam_size = beam_size
                    logger.info(f"Updated beam_size to {beam_size}")
                except Exception as e:
                    logger.warning(f"Failed to update beam_size: {e}")
        
        # Also update transcription parameters dict
        if hasattr(self, 'transcribe_params'):
            if temperature is not None:
                self.transcribe_params['temperature'] = temperature
            if initial_prompt is not None:
                self.transcribe_params['initial_prompt'] = initial_prompt
            if max_tokens is not None and 'max_tokens' in self.transcribe_params:
                self.transcribe_params['max_tokens'] = max_tokens

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
        
        # Store the parameters
        self.temperature = preset_params["temperature"]
        self.initial_prompt = preset_params["initial_prompt"]
        self.max_tokens = preset_params["max_tokens"]
        self.no_context = preset_params["no_context"]
        self.single_segment = preset_params["single_segment"]
        
        # Update the transcribe params
        self.transcribe_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "initial_prompt": self.initial_prompt
        }
        
        # Apply the parameters to the model if possible
        if self.can_set_temperature:
            try:
                self.model.temperature = self.temperature
            except Exception as e:
                logger.warning(f"Could not set temperature: {e}")
                
        if self.can_set_no_context:
            try:
                self.model.no_context = self.no_context
            except Exception as e:
                logger.warning(f"Could not set no_context: {e}")
                
        if self.can_set_single_segment:
            try:
                self.model.single_segment = self.single_segment
            except Exception as e:
                logger.warning(f"Could not set single_segment: {e}")
                
        if self.can_set_initial_prompt:
            try:
                self.model.initial_prompt = self.initial_prompt
            except Exception as e:
                logger.warning(f"Could not set initial_prompt: {e}")

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

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio to enhance speech and reduce noise.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Preprocessed audio data
        """
        try:
            from scipy import signal
            
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            b, a = signal.butter(4, 80/(self.sample_rate/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # 2. Apply light noise reduction
            noise_threshold = 0.005  # Very low amplitude is likely noise
            noise_gate = np.where(np.abs(filtered_audio) < noise_threshold, 0, filtered_audio)
            
            # 3. Normalize audio to have consistent volume
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
                
            return normalized
        except ImportError:
            logger.warning("SciPy not available, skipping audio preprocessing")
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio  # Return original if preprocessing fails

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

        # Ensure audio is in range [-1.0, 1.0]
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
        
        # Apply audio preprocessing for noise reduction
        audio_chunk = self._preprocess_audio(audio_chunk)

        # Add to input buffer
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

        # Rough estimate of start/end times
        chunk_start_time = current_stream_time
        chunk_end_time = chunk_start_time + chunk_duration

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            start_sample=int(chunk_start_time * self.sample_rate),
            end_sample=int(chunk_end_time * self.sample_rate),
            sample_rate=self.sample_rate,
            is_first_chunk=(chunk_id == 1),
        )

        # Perform voice activity detection if enabled
        contains_speech = True
        if self.vad_enabled:
            contains_speech = self._detect_speech(chunk, threshold=self.chunker.silence_threshold)

        # If no speech detected, skip transcription
        if not contains_speech:
            logger.debug(f"No speech detected in chunk {chunk_id}, skipping transcription")
            result = StreamingTranscriptionResult(
                text="",
                is_final=True,
                confidence=0.0,
                start_time=chunk_start_time,
                end_time=chunk_end_time,
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
            
            # Clean up the transcription
            combined_text = self.cleanup_transcription(combined_text)
            
            # If no meaningful text after cleanup, return None
            if not combined_text:
                logger.debug(f"No meaningful text after cleanup for chunk {chunk_id}")
                return None

            # Estimate confidence (pywhispercpp doesn't provide it directly)
            # We can estimate it based on the presence of uncertainty markers
            confidence = 1.0
            if "[" in combined_text or "(" in combined_text or "?" in combined_text:
                confidence = 0.6  # Lower confidence for uncertain transcriptions
                logger.debug(f"Reduced confidence due to uncertainty markers: {combined_text}")

            # Create streaming result
            result = StreamingTranscriptionResult(
                text=combined_text,
                is_final=True,  # For now, all results are final per chunk
                confidence=confidence,
                start_time=chunk_start_time,
                end_time=chunk_end_time,
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

    def _safe_transcribe(self, audio_data):
        """
        Safely transcribe audio data with improved noise-optimized parameters.
        """
        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Handle short audio
        min_audio_length_samples = int(self.sample_rate * 0.1)
        if len(audio_data) < min_audio_length_samples:
            # Pad with silence if too short
            required_padding = min_audio_length_samples - len(audio_data)
            logger.info(f"Audio too short ({len(audio_data)/self.sample_rate:.3f}s), padding to {min_audio_length_samples/self.sample_rate:.3f}s")
            padding = np.zeros(required_padding, dtype=np.float32)
            audio_data = np.concatenate([audio_data, padding])

        # Set core parameters safely
        if self.can_set_temperature:
            try:
                self.model.temperature = self.temperature
            except Exception as e:
                 logger.warning(f"Failed to set temperature dynamically: {e}")
                 self.can_set_temperature = False

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
                 
        # Set initial prompt if available
        if self.can_set_initial_prompt and self.initial_prompt:
            try:
                self.model.initial_prompt = self.initial_prompt
            except Exception as e:
                logger.warning(f"Failed to set initial_prompt dynamically: {e}")
                self.can_set_initial_prompt = False
        
        # Additional parameters specifically for noise handling
        # Note: These might not be available in all versions of pywhispercpp
        try:
            # Set beam search parameters if available
            if self.can_set_beam_size:
                self.model.beam_size = 3  # More potential hypotheses
                
            if self.can_set_patience:
                self.model.patience = 1.0  # Wait longer to find better matches
                
            # Set higher entropy threshold to avoid forcing transcription when unsure
            if self.can_set_entropy_threshold:
                self.model.entropy_threshold = 2.8  # Higher than default
            
            # Ignore initial silence - helps with false triggers
            if self.can_set_suppress_blank:
                self.model.suppress_blank = True
                
        except Exception as e:
            logger.warning(f"Failed to set advanced parameters: {e}")

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

    def _detect_speech(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Enhanced voice activity detection based on multiple features.

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
        
        # Calculate zero-crossing rate (helps distinguish speech from noise)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
        
        # Try to calculate spectral centroid (speech typically has higher centroids than noise)
        try:
            fft_data = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            spectral_centroid = np.sum(freqs * fft_data) / (np.sum(fft_data) + 1e-10)
        except Exception as e:
            logger.warning(f"Couldn't calculate spectral centroid: {e}")
            spectral_centroid = 1000  # Default to a neutral value
        
        # Check for frequency distribution (speech typically has significant mid-range energy)
        try:
            fft_data = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            
            # Split frequency range into low, mid, high bands
            low_freq_idx = (freqs < 200)
            mid_freq_idx = (freqs >= 200) & (freqs <= 3000)
            high_freq_idx = (freqs > 3000)
            
            # Calculate energy in each band
            low_energy = np.sum(fft_data[low_freq_idx])
            mid_energy = np.sum(fft_data[mid_freq_idx])
            high_energy = np.sum(fft_data[high_freq_idx])
            
            total_energy = low_energy + mid_energy + high_energy + 1e-10
            
            # Calculate ratios
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            
            # Speech typically has significant mid-range energy
            good_speech_distribution = mid_ratio > 0.3
        except Exception as e:
            logger.warning(f"Couldn't calculate frequency distribution: {e}")
            good_speech_distribution = True  # Default
        
        # Log detailed VAD information at debug level
        logger.debug(f"VAD metrics - RMS: {rms:.6f}, ZCR: {zero_crossings:.6f}, " 
                     f"Centroid: {spectral_centroid:.1f}Hz, Mid-ratio: {mid_ratio:.2f}")
        
        # Combined decision using multiple features
        is_speech = (rms > threshold and                  # Energy above threshold
                    zero_crossings > 0.01 and             # Some zero-crossings (not a pure tone/noise)
                    zero_crossings < 0.15 and             # Not too many zero-crossings (not white noise)
                    spectral_centroid > 300 and           # Speech tends to have higher centroids
                    good_speech_distribution)             # Good distribution of frequency energies
        
        return is_speech

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