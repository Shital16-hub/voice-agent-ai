"""
Enhanced Speech-to-Text integration module for Voice AI Agent with improved noise handling
and foreground speech extraction.

This module provides classes and functions for integrating Google Cloud speech-to-text
capabilities with the Voice AI Agent system.
"""
import logging
import time
import asyncio
import re
import numpy as np
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator
from scipy import signal
from collections import deque

from speech_to_text.google_stt import GoogleCloudSTT, STTStreamer, SpeechDetector
from speech_to_text.streamer import GoogleCloudStreamer, StreamingTranscriptionResult
from speech_to_text.utils.audio_utils import load_audio_file

# Continue with the rest of the file...

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

# Speech state enum for state machine
class SpeechState:
    """Speech state for detection state machine"""
    SILENCE = 0
    POTENTIAL_SPEECH = 1
    CONFIRMED_SPEECH = 2
    SPEECH_ENDED = 3

class STTIntegration:
    """
    Enhanced Speech-to-Text integration for Voice AI Agent with Deepgram.
    
    Provides an abstraction layer for speech recognition functionality,
    handling audio processing and transcription with improved speech/noise discrimination.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamer] = None,
        language: str = "en"
    ):
        """
        Initialize the STT integration.
        
        Args:
            speech_recognizer: Initialized GoogleCloudStreamer instance
            language: Language code for speech recognition
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Enhanced adaptive noise floor tracking
        self.noise_samples = []
        self.max_samples = 30  # Increased from 20 for better statistics
        self.ambient_noise_level = 0.01  # Starting threshold
        self.min_noise_floor = 0.005  # Minimum noise floor
        
        # Enhanced speech detection with state machine
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        self.speech_threshold = 0.015  # Initial threshold
        
        # Speech frequency band energy tracking
        self.speech_band_energies = deque(maxlen=20)
        
        # Energy thresholds with hysteresis
        self.low_threshold = self.ambient_noise_level * 2.0  # For detecting potential speech
        self.high_threshold = self.ambient_noise_level * 3.5  # For confirming speech
    
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
            self.speech_recognizer = GoogleCloudStreamer(
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
        Update ambient noise level using adaptive statistics.
        
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
                # Use 90th percentile to avoid outliers
                self.ambient_noise_level = max(
                    self.min_noise_floor,  # Minimum threshold
                    np.percentile(self.noise_samples, 90) * 2.0  # Set threshold just above noise
                )
                logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
                
                # Update derived thresholds
                self.low_threshold = self.ambient_noise_level * 2.0
                self.high_threshold = self.ambient_noise_level * 3.5
    
    def _apply_spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction to reduce background noise.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Enhanced audio data
        """
        if len(audio_data) < 512:  # Need enough samples for a good FFT
            return audio_data
            
        try:
            # Compute STFT
            f, t, Zxx = signal.stft(audio_data, fs=16000, nperseg=512, noverlap=384)
            
            # Compute magnitude spectrum
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Estimate noise spectrum from lowest 10% of frame energies
            frame_energy = np.sum(magnitude**2, axis=0)
            sorted_frames = np.argsort(frame_energy)
            noise_frames = max(1, int(0.1 * len(sorted_frames)))
            noise_indices = sorted_frames[:noise_frames]
            
            # Average noise spectrum over noise frames
            noise_spectrum = np.mean(magnitude[:, noise_indices], axis=1, keepdims=True)
            
            # Apply spectral subtraction with flooring
            # Use over-subtraction factor for better noise reduction
            over_subtraction = 1.5
            spectral_floor = 0.01
            
            # Apply subtraction with flooring
            magnitude_enhanced = np.maximum(
                magnitude - over_subtraction * noise_spectrum,
                spectral_floor * magnitude
            )
            
            # Reconstruct signal
            Zxx_enhanced = magnitude_enhanced * np.exp(1j * phase)
            _, audio_enhanced = signal.istft(Zxx_enhanced, fs=16000, nperseg=512, noverlap=384)
            
            # Ensure same length as original
            if len(audio_enhanced) < len(audio_data):
                audio_enhanced = np.pad(audio_enhanced, (0, len(audio_data) - len(audio_enhanced)))
            elif len(audio_enhanced) > len(audio_data):
                audio_enhanced = audio_enhanced[:len(audio_data)]
            
            return audio_enhanced
            
        except Exception as e:
            logger.error(f"Error in spectral subtraction: {e}")
            return audio_data  # Return original if processing fails
    
    def _calculate_speech_band_energy(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate energy in specific frequency bands relevant to speech.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with energy in different frequency bands
        """
        if len(audio_data) < 256:  # Need enough samples for FFT
            return {
                "low_band": 0.0,
                "speech_band": 0.0,
                "high_band": 0.0,
                "speech_ratio": 0.0
            }
            
        try:
            # Calculate FFT
            fft_data = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1/16000)
            
            # Define frequency bands
            low_band_idx = (freqs < 300)
            speech_band_idx = (freqs >= 300) & (freqs <= 3400)  # Primary speech frequencies
            high_band_idx = (freqs > 3400)
            
            # Calculate energy in each band
            low_energy = np.sum(fft_data[low_band_idx])
            speech_energy = np.sum(fft_data[speech_band_idx])
            high_energy = np.sum(fft_data[high_band_idx])
            
            total_energy = low_energy + speech_energy + high_energy + 1e-10
            
            # Calculate ratios
            speech_ratio = speech_energy / total_energy
            
            result = {
                "low_band": low_energy / total_energy,
                "speech_band": speech_ratio,
                "high_band": high_energy / total_energy,
                "speech_ratio": speech_ratio
            }
            
            # Store speech band energy for tracking
            self.speech_band_energies.append(speech_ratio)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating frequency bands: {e}")
            return {
                "low_band": 0.0,
                "speech_band": 0.0,
                "high_band": 0.0,
                "speech_ratio": 0.0
            }
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhanced multi-stage audio preprocessing for better speech/noise discrimination.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Preprocessed audio data
        """
        try:
            # Update ambient noise level
            self._update_ambient_noise_level(audio_data)
            
            # 1. Apply high-pass filter to remove low-frequency noise (below 100Hz)
            b, a = signal.butter(6, 100/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply band-pass filter for telephony frequency range (300-3400 Hz)
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            band_limited = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Apply spectral subtraction for background noise reduction
            if len(band_limited) >= 512:
                noise_reduced = self._apply_spectral_subtraction(band_limited)
            else:
                noise_reduced = band_limited
            
            # 4. Apply adaptive noise gate with threshold based on ambient noise
            threshold = self.ambient_noise_level * 2.5
            noise_gate = np.where(np.abs(noise_reduced) < threshold, 0, noise_reduced)
            
            # 5. Apply pre-emphasis filter to boost higher frequencies for speech clarity
            pre_emphasis = np.append(noise_gate[0], noise_gate[1:] - 0.97 * noise_gate[:-1])
            
            # 6. Apply speech onset enhancement - boost sudden changes in energy
            if len(pre_emphasis) > 320:  # At least 20ms
                # Calculate energy envelope
                frame_size = 160  # 10ms
                energy_envelope = np.array([
                    np.mean(np.square(pre_emphasis[i:i+frame_size]))
                    for i in range(0, len(pre_emphasis)-frame_size, frame_size)
                ])
                
                # Compute derivative of energy
                energy_derivative = np.diff(energy_envelope, prepend=energy_envelope[0])
                
                # Find regions of rising energy (speech onset)
                rising_energy = energy_derivative > 0
                
                # Expand to full audio length
                rising_energy_full = np.repeat(rising_energy, frame_size)
                if len(rising_energy_full) < len(pre_emphasis):
                    rising_energy_full = np.pad(
                        rising_energy_full,
                        (0, len(pre_emphasis) - len(rising_energy_full)),
                        'edge'
                    )
                
                # Apply gentle boost to regions of rising energy
                boost_factor = np.ones_like(pre_emphasis)
                boost_factor[rising_energy_full] = 1.2  # 20% boost for speech onsets
                pre_emphasis = pre_emphasis * boost_factor
            
            # 7. Normalize audio level for consistent volume
            max_val = np.max(np.abs(pre_emphasis))
            if max_val > 0:
                normalized = pre_emphasis * (0.9 / max_val)
            else:
                normalized = pre_emphasis
                
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
        Check if a transcription is valid and worth processing with enhanced criteria.
        
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
        
        # Be more lenient with question marks and punctuation in confidence calculation
        confidence_estimate = 1.0
        if "?" in text:
            # Questions are important in conversations - don't penalize them
            pass
        elif "[" in text or "(" in text or "<" in text:
            confidence_estimate = 0.7  # Only reduce for annotation markers
            logger.info(f"Reduced confidence due to uncertainty markers: {text}")
            
        if confidence_estimate < 0.65:
            logger.info(f"Transcription confidence too low: {confidence_estimate}")
            return False
            
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < min_words:
            logger.info(f"Transcription too short: {word_count} words")
            return False
            
        return True
    
    def _contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced speech detection with state machine for better noise discrimination.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if the audio contains speech
        """
        if len(audio_data) < 500:  # Need enough samples
            return False
            
        try:
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(np.square(audio_data)))
            
            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
            
            # Calculate frequency band energies
            band_energies = self._calculate_speech_band_energy(audio_data)
            speech_ratio = band_energies["speech_ratio"]
            
            # Calculate spectral flux - measure of how quickly spectrum changes
            if len(audio_data) >= 512:
                half_point = len(audio_data) // 2
                first_half = audio_data[:half_point]
                second_half = audio_data[half_point:2*half_point]
                
                # Calculate spectra
                first_spectrum = np.abs(np.fft.rfft(first_half))
                second_spectrum = np.abs(np.fft.rfft(second_half))
                
                # Calculate normalized spectral flux
                if np.sum(first_spectrum) > 0 and np.sum(second_spectrum) > 0:
                    # Normalize spectra
                    first_spectrum = first_spectrum / np.sum(first_spectrum)
                    second_spectrum = second_spectrum / np.sum(second_spectrum)
                    
                    # Calculate flux (sum of squared differences)
                    spectral_flux = np.sum(np.square(second_spectrum - first_spectrum))
                else:
                    spectral_flux = 0.0
            else:
                spectral_flux = 0.0
            
            # Get adaptive thresholds
            energy_threshold = self.high_threshold
            low_energy_threshold = self.low_threshold
            
            # Calculate the average speech band ratio from history
            avg_speech_ratio = 0.4  # Default
            if self.speech_band_energies:
                avg_speech_ratio = np.mean(self.speech_band_energies)
            
            # State machine for more robust speech detection
            if self.speech_state == SpeechState.SILENCE:
                # Check if potential speech detected
                if rms_energy > low_energy_threshold and speech_ratio > 0.5:
                    self.speech_state = SpeechState.POTENTIAL_SPEECH
                    self.potential_speech_frames = 1
                    logger.debug(f"Potential speech detected: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                    return False  # Not confirmed yet
                return False  # Still silence
                
            elif self.speech_state == SpeechState.POTENTIAL_SPEECH:
                # Check if still potential speech
                if rms_energy > low_energy_threshold and speech_ratio > 0.5:
                    self.potential_speech_frames += 1
                    # Check if we have enough frames to move to confirmed speech
                    if self.potential_speech_frames >= 3:  # Need 3 consecutive frames
                        # Check against higher threshold for confirmation
                        if rms_energy > energy_threshold and speech_ratio > 0.6:
                            self.speech_state = SpeechState.CONFIRMED_SPEECH
                            logger.info(f"Speech confirmed: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                            return True
                    return False  # Not confirmed yet
                else:
                    # Go back to silence
                    self.speech_state = SpeechState.SILENCE
                    self.potential_speech_frames = 0
                    return False
                    
            elif self.speech_state == SpeechState.CONFIRMED_SPEECH:
                # Check if still speech
                if rms_energy > energy_threshold * 0.8 or speech_ratio > 0.4:
                    # Still speech, maintain state
                    return True
                else:
                    # Potential end of speech
                    self.speech_state = SpeechState.SPEECH_ENDED
                    self.confirmed_speech_frames = 1
                    return True  # Still report as speech
                    
            elif self.speech_state == SpeechState.SPEECH_ENDED:
                # Check if silence is confirmed
                if rms_energy < energy_threshold or speech_ratio < 0.4:
                    self.confirmed_speech_frames += 1
                    # Check if we have enough frames to confirm end of speech
                    if self.confirmed_speech_frames >= 3:  # Need 3 consecutive frames
                        self.speech_state = SpeechState.SILENCE
                        self.confirmed_speech_frames = 0
                        return False
                    return True  # Still report as speech until confirmed end
                else:
                    # Speech resumed
                    self.speech_state = SpeechState.CONFIRMED_SPEECH
                    self.confirmed_speech_frames = 0
                    return True
                    
            # Default fallback - use direct detection
            is_speech = (
                (rms_energy > energy_threshold) and 
                (zero_crossings > 0.01 and zero_crossings < 0.15) and
                ((speech_ratio > 0.5 and speech_ratio > avg_speech_ratio * 0.8) or spectral_flux > 0.2)
            )
            
            if is_speech:
                logger.debug("Speech detected through direct conditions")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            # Fall back to simple energy threshold
            energy = np.mean(np.abs(audio_data))
            return energy > self.high_threshold
    
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
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
            audio_duration = len(audio) / sample_rate
            
            # Apply enhanced preprocessing
            audio = self._preprocess_audio(audio)
            
            # Check if audio contains speech
            contains_speech = self._contains_speech(audio)
            if not contains_speech:
                logger.info(f"No speech detected in audio file: {audio_file_path}")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": audio_duration,
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": False
                }
            
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
        Transcribe audio data with enhanced speech processing.
        
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
            
            # Apply enhanced processing
            audio = self._preprocess_audio(audio)
            
            # Check if audio contains speech
            contains_speech = self._contains_speech(audio)
            if not contains_speech:
                logger.info("No speech detected in audio data")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": len(audio) / 16000,  # Assuming 16kHz
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": False
                }
            
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
        
        # Reset speech detection state
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
    
    async def process_stream_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray, List[float]],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a chunk of streaming audio with enhanced speech processing.
        
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
        
        # Apply enhanced processing
        audio_data = self._preprocess_audio(audio_data)
        
        # Only process further if speech is detected
        # Use a lightweight check for streaming to reduce latency
        if not self._contains_speech(audio_data) and self.speech_state == SpeechState.SILENCE:
            # Skip processing if no speech detected in silence state
            # Still allow processing in other states to continue tracking ongoing speech
            return None
        
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
        
        # Reset speech detection state
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        
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
        Process real-time audio stream and detect utterances with enhanced speech processing.
        
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
        
        # Reset speech detection state
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        
        # Track consecutive silence frames
        silence_frames = 0
        max_silence_frames = silence_frames_threshold
        
        try:
            async for audio_chunk in audio_stream:
                # Apply enhanced processing
                processed_chunk = self._preprocess_audio(audio_chunk)
                
                # Check for speech presence
                contains_speech = self._contains_speech(processed_chunk)
                
                # Convert to bytes (16-bit PCM)
                audio_bytes = (processed_chunk * 32767).astype(np.int16).tobytes()
                
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
                
                # Update silence frame tracking based on speech detection
                if not contains_speech or self.speech_state == SpeechState.SILENCE:
                    silence_frames += 1
                else:
                    silence_frames = 0
                
                # If we've detected enough silence after speech, process the utterance
                if (self.speech_state != SpeechState.SILENCE and silence_frames >= max_silence_frames):
                    # Reset speech state
                    self.speech_state = SpeechState.SILENCE
                    self.potential_speech_frames = 0
                    self.confirmed_speech_frames = 0
                    
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