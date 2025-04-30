"""
Voice AI Agent main class that coordinates all components with Deepgram STT integration.
Enhanced with improved speech/noise discrimination for telephony applications.
Generic version that works with any knowledge base.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np
from scipy import signal
from collections import deque

# Deepgram STT imports
from speech_to_text.deepgram_stt import DeepgramStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import GoogleCloudTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """
    Main Voice AI Agent class with enhanced speech/noise discrimination and foreground
    speech extraction for telephony applications.
    """
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        api_key: Optional[str] = None,
        llm_temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with Deepgram STT and enhanced speech processing.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            api_key: Deepgram API key (defaults to env variable)
            llm_temperature: LLM temperature for response generation
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.api_key = api_key
        self.llm_temperature = llm_temperature
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        self.stt_keywords = kwargs.get('keywords', ['price', 'plan', 'cost', 'subscription', 'service'])
        
        # Enhanced speech processing parameters
        self.whisper_initial_prompt = kwargs.get('whisper_initial_prompt', 
            "This is a clear business conversation. Transcribe the exact words spoken, ignoring background noise.")
        self.whisper_temperature = kwargs.get('whisper_temperature', 0.0)
        self.whisper_no_context = kwargs.get('whisper_no_context', True)
        self.whisper_preset = kwargs.get('whisper_preset', "default")
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Enhanced adaptive noise tracking
        self.noise_floor = 0.005  # Initial estimate
        self.noise_samples = []
        self.max_noise_samples = 30  # Increased from 20 for better statistics
        
        # Speech/noise analysis history
        self.speech_band_energies = deque(maxlen=20)
        self.noise_band_energies = deque(maxlen=20)
        self.speech_to_noise_ratios = deque(maxlen=10)
        
        # Speech detection state machine
        self.speech_state = 0  # 0=silence, 1=potential speech, 2=confirmed speech, 3=speech ended
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        
    def _update_noise_floor(self, audio: np.ndarray) -> None:
        """
        Update noise floor estimate from quiet sections of audio.
        Uses statistical analysis for more robust adaptation.
        
        Args:
            audio: Audio data as numpy array
        """
        # Find quiet sections (bottom 10% of energy)
        frame_size = min(len(audio), int(0.02 * 16000))  # 20ms frames
        if frame_size <= 1:
            return
            
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        frame_energies = [np.mean(np.square(frame)) for frame in frames]
        
        if len(frame_energies) > 0:
            # Sort energies and take bottom 10%
            sorted_energies = sorted(frame_energies)
            quiet_count = max(1, len(sorted_energies) // 10)
            quiet_energies = sorted_energies[:quiet_count]
            
            # Update noise samples
            self.noise_samples.extend(quiet_energies)
            
            # Limit sample count
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples = self.noise_samples[-self.max_noise_samples:]
            
            # Update noise floor with safety limits
            if self.noise_samples:
                self.noise_floor = max(
                    0.001,  # Minimum
                    min(0.02, np.percentile(self.noise_samples, 90) * 1.5)  # Maximum
                )
                logger.debug(f"Updated noise floor to {self.noise_floor:.6f}")
    
    def _calculate_speech_band_energy(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Calculate energy in specific frequency bands relevant to speech.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with energy in different frequency bands
        """
        if len(audio) < 256:  # Need enough samples for FFT
            return {
                "low_band": 0.0,
                "speech_band": 0.0,
                "high_band": 0.0,
                "speech_ratio": 0.0
            }
            
        try:
            # Calculate FFT
            fft_data = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/16000)
            
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
            
            # Track speech band energy
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
    
    def _apply_spectral_subtraction(self, audio: np.ndarray, frame_size: int = 512, overlap: int = 384) -> np.ndarray:
        """
        Apply spectral subtraction to reduce background noise.
        
        Args:
            audio: Audio data as numpy array
            frame_size: Frame size for STFT
            overlap: Overlap between frames
            
        Returns:
            Enhanced audio data
        """
        try:
            # Compute Short-Time Fourier Transform (STFT)
            f, t, Zxx = signal.stft(audio, fs=16000, nperseg=frame_size, noverlap=overlap)
            
            # Get magnitude and phase components
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Estimate noise spectrum from lowest energy frames
            mag_energy = np.sum(mag**2, axis=0)
            sorted_indices = np.argsort(mag_energy)
            
            # Use first 10% of frames or at least 1 frame as noise estimate
            noise_frames = max(1, int(0.1 * len(sorted_indices)))
            noise_indices = sorted_indices[:noise_frames]
            
            # Compute noise estimate for each frequency bin
            noise_estimate = np.mean(mag[:, noise_indices], axis=1, keepdims=True)
            
            # Apply spectral subtraction with flooring to avoid negative values
            # and oversubtraction to reduce musical noise
            oversubtraction_factor = 1.5  # Subtract more noise than estimated
            spectral_floor = 0.01  # Minimum magnitude after subtraction
            
            # Apply subtraction with flooring
            mag_enhanced = np.maximum(
                mag - oversubtraction_factor * noise_estimate,
                spectral_floor * mag
            )
            
            # Reconstruct complex spectrum with original phase
            Zxx_enhanced = mag_enhanced * np.exp(1j * phase)
            
            # Inverse STFT to get time domain signal
            _, audio_enhanced = signal.istft(Zxx_enhanced, fs=16000, nperseg=frame_size, noverlap=overlap)
            
            # Ensure same length as original signal
            if len(audio_enhanced) < len(audio):
                # Zero-pad if shorter
                audio_enhanced = np.pad(audio_enhanced, (0, len(audio) - len(audio_enhanced)))
            elif len(audio_enhanced) > len(audio):
                # Truncate if longer
                audio_enhanced = audio_enhanced[:len(audio)]
            
            return audio_enhanced
            
        except Exception as e:
            logger.error(f"Error applying spectral subtraction: {e}")
            return audio  # Return original signal if error
            
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhanced multi-stage audio processing for improved speech/noise discrimination.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        try:
            # Update noise floor from quiet sections
            self._update_noise_floor(audio)
            
            # Store original for potential mixing in output
            original_audio = audio.copy()
            
            # 1. Apply high-pass filter to remove low-frequency noise (below 100Hz)
            b, a = signal.butter(6, 100/(16000/2), 'highpass')
            audio = signal.filtfilt(b, a, audio)
            
            # 2. Apply band-pass filter for telephony frequency range (300-3400 Hz)
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            audio = signal.filtfilt(b, a, audio)
            
            # 3. Apply spectral subtraction for noise reduction
            if len(audio) >= 512:
                audio = self._apply_spectral_subtraction(audio)
            
            # 4. Apply speech onset emphasis - boost sudden changes in energy
            if len(audio) > 320:  # At least 20ms
                # Calculate energy envelope
                frame_size = 160  # 10ms
                energy_envelope = np.array([
                    np.mean(np.square(audio[i:i+frame_size]))
                    for i in range(0, len(audio)-frame_size, frame_size)
                ])
                
                # Compute derivative of energy
                energy_derivative = np.diff(energy_envelope, prepend=energy_envelope[0])
                
                # Find regions of rising energy (speech onset)
                rising_energy = energy_derivative > 0
                
                # Expand to full audio length
                rising_energy_full = np.repeat(rising_energy, frame_size)
                if len(rising_energy_full) < len(audio):
                    rising_energy_full = np.pad(
                        rising_energy_full,
                        (0, len(audio) - len(rising_energy_full)),
                        'edge'
                    )
                
                # Apply gentle boost to regions of rising energy
                boost_factor = np.ones_like(audio)
                boost_factor[rising_energy_full] = 1.2  # 20% boost for speech onsets
                audio = audio * boost_factor
            
            # 5. Apply adaptive noise gate
            threshold = max(0.015, self.noise_floor * 3.0)
            audio = np.where(np.abs(audio) < threshold, 0, audio)
            
            # 6. Apply pre-emphasis to boost high frequencies (speech clarity)
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
            # 7. Normalize the audio level
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * (0.9 / max_val)
                
            # 8. Use speech detection to possibly blend with original for natural sound
            # Calculate speech band energy to guide the process
            band_energies = self._calculate_speech_band_energy(audio)
            speech_ratio = band_energies["speech_ratio"]
            
            # If strong speech characteristics, keep processed audio
            # If weak speech but still some, blend processed and original
            # This prevents over-processing of faint speech segments
            if speech_ratio > 0.6:
                # Strong speech - keep enhanced version
                return audio
            elif speech_ratio > 0.4:
                # Moderate speech - blend processed and original
                blend_factor = (speech_ratio - 0.4) / 0.2  # Maps 0.4-0.6 to 0.0-1.0
                blended = blend_factor * audio + (1 - blend_factor) * original_audio
                return blended
            else:
                # Not very speech-like - might be silence or noise
                # For silence, return zeros; for noise use heavy processing
                if np.mean(np.abs(audio)) < self.noise_floor * 2:
                    # Silence - further reduce
                    return audio * 0.1
                else:
                    # Noise - use heavily processed version
                    return audio
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio  # Return original if processing fails
    
    def _contains_speech(self, audio: np.ndarray) -> bool:
        """
        Enhanced speech detection with state machine for better discrimination.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            True if audio contains speech
        """
        if len(audio) < 500:  # Need enough samples
            return False
            
        try:
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(np.square(audio)))
            
            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
            
            # Calculate frequency band energies
            band_energies = self._calculate_speech_band_energy(audio)
            speech_ratio = band_energies["speech_ratio"]
            
            # Calculate spectral flux - measure of how quickly spectrum changes
            if len(audio) >= 512:
                half_point = len(audio) // 2
                first_half = audio[:half_point]
                second_half = audio[half_point:2*half_point]
                
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
            energy_threshold = max(0.015, self.noise_floor * 3.0)
            
            # Calculate the average speech band ratio from history
            avg_speech_ratio = 0.4  # Default
            if self.speech_band_energies:
                avg_speech_ratio = np.mean(self.speech_band_energies)
            
            # State machine for more robust speech detection
            # State 0: Silence - waiting for potential speech
            # State 1: Potential speech - need confirmation
            # State 2: Confirmed speech - tracking speech
            # State 3: Speech ending - waiting to confirm end
            
            if self.speech_state == 0:  # Silence
                # Check if potential speech detected
                if rms_energy > energy_threshold and speech_ratio > 0.5:
                    self.speech_state = 1  # Potential speech
                    self.speech_frame_count = 1
                    logger.debug(f"Potential speech detected: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                    return False  # Not confirmed yet
                return False  # Still silence
                
            elif self.speech_state == 1:  # Potential speech
                # Check if still potential speech
                if rms_energy > energy_threshold and speech_ratio > 0.5:
                    self.speech_frame_count += 1
                    # Check if we have enough frames to confirm speech
                    if self.speech_frame_count >= 3:  # Need 3 consecutive frames
                        self.speech_state = 2  # Confirmed speech
                        logger.info(f"Speech confirmed: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                        return True
                    return False  # Not confirmed yet
                else:
                    # Go back to silence
                    self.speech_state = 0
                    self.speech_frame_count = 0
                    return False
                    
            elif self.speech_state == 2:  # Confirmed speech
                # Check if still speech
                if rms_energy > energy_threshold * 0.8 and speech_ratio > 0.4:
                    # Still speech, maintain state
                    return True
                else:
                    # Potential end of speech
                    self.speech_state = 3
                    self.silence_frame_count = 1
                    return True  # Still report as speech
                    
            elif self.speech_state == 3:  # Speech ending
                # Check if silence is confirmed
                if rms_energy < energy_threshold or speech_ratio < 0.4:
                    self.silence_frame_count += 1
                    # Check if we have enough frames to confirm end of speech
                    if self.silence_frame_count >= 3:  # Need 3 consecutive frames
                        self.speech_state = 0  # Back to silence
                        self.silence_frame_count = 0
                        return False
                    return True  # Still report as speech until confirmed end
                else:
                    # Speech resumed
                    self.speech_state = 2
                    self.silence_frame_count = 0
                    return True
                    
            # Default fallback - use direct detection
            direct_speech = (
                (rms_energy > energy_threshold) and 
                (zero_crossings > 0.01 and zero_crossings < 0.15) and
                ((speech_ratio > 0.5 and speech_ratio > avg_speech_ratio * 0.8) or spectral_flux > 0.2)
            )
            
            if direct_speech:
                logger.debug("Speech detected through direct conditions")
            
            return direct_speech
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            # Fall back to simple energy threshold
            energy = np.mean(np.abs(audio))
            return energy > max(0.015, self.noise_floor * 3.0)
            
    async def init(self):
        """Initialize all components with Deepgram STT."""
        logger.info("Initializing Voice AI Agent components with enhanced speech processing...")
        
        # Initialize speech recognizer with Deepgram
        self.speech_recognizer = DeepgramStreamingSTT(
            api_key=self.api_key,
            language=self.stt_language,
            sample_rate=16000,
            encoding="linear16",
            channels=1,
            interim_results=True
        )
        
        # Initialize STT integration 
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.stt_language
        )
        
        # Initialize document store and index manager
        doc_store = DocumentStore()
        index_manager = IndexManager(storage_dir=self.storage_dir)
        await index_manager.init()
        
        # Initialize query engine
        self.query_engine = QueryEngine(
            index_manager=index_manager, 
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature
        )
        await self.query_engine.init()
        
        # Initialize conversation manager with optimized parameters
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            # Skip greeting for better telephone experience
            skip_greeting=True
        )
        await self.conversation_manager.init()
        
        # Initialize TTS client
        self.tts_client = GoogleCloudTTS()
        
        logger.info("Voice AI Agent initialization complete with enhanced speech processing")
        
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with enhanced speech/noise discrimination.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Process audio for better speech recognition
        if isinstance(audio_data, np.ndarray):
            audio_data = self._process_audio(audio_data)
            
            # Check if audio contains actual speech
            contains_speech = self._contains_speech(audio_data)
            if not contains_speech:
                logger.info("No speech detected in audio, skipping processing")
                return {
                    "status": "no_speech",
                    "transcription": "",
                    "error": "No speech detected"
                }
        
        # Use STT integration for processing with Deepgram
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(transcription)
            
            return {
                "transcription": transcription,
                "response": response.get("response", ""),
                "status": "success"
            }
        else:
            logger.info("Invalid or empty transcription")
            return {
                "status": "invalid_transcription",
                "transcription": result.get("transcription", ""),
                "error": "No valid speech detected"
            }
    
    async def process_streaming_audio(
        self,
        audio_stream,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process streaming audio with real-time response and enhanced speech processing.
        
        Args:
            audio_stream: Async iterator of audio chunks
            result_callback: Callback for streaming results
            
        Returns:
            Final processing stats
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Track stats
        start_time = time.time()
        chunks_processed = 0
        results_count = 0
        
        # Start streaming session
        await self.speech_recognizer.start_streaming()
        
        try:
            # Process each audio chunk
            async for chunk in audio_stream:
                chunks_processed += 1
                
                # Process audio for better recognition if it's numpy array
                if isinstance(chunk, np.ndarray):
                    chunk = self._process_audio(chunk)
                    
                    # Skip further processing if no speech detected
                    if not self._contains_speech(chunk) and chunks_processed % 5 != 0:
                        # Only skip some chunks to ensure we don't miss speech onset
                        continue
                
                # Convert to bytes for Deepgram if needed
                if isinstance(chunk, np.ndarray):
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = chunk
                    
                # Process through Deepgram
                async def process_result(result):
                    # Only handle final results
                    if result.is_final:
                        # Clean up transcription
                        transcription = self.stt_integration.cleanup_transcription(result.text)
                        
                        # Process if valid
                        if transcription and self.stt_integration.is_valid_transcription(transcription):
                            # Get response from conversation manager
                            response = await self.conversation_manager.handle_user_input(transcription)
                            
                            # Format result
                            result_data = {
                                "transcription": transcription,
                                "response": response.get("response", ""),
                                "confidence": result.confidence,
                                "is_final": True
                            }
                            
                            nonlocal results_count
                            results_count += 1
                            
                            # Call callback if provided
                            if result_callback:
                                await result_callback(result_data)
                
                # Process chunk
                await self.speech_recognizer.process_audio_chunk(audio_bytes, process_result)
                
            # Stop streaming session
            await self.speech_recognizer.stop_streaming()
            
            # Return stats
            return {
                "status": "complete",
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming audio: {e}")
            
            # Stop streaming session
            await self.speech_recognizer.stop_streaming()
            
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return (self.speech_recognizer is not None and 
                self.conversation_manager is not None and 
                self.query_engine is not None)
                
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Close Deepgram streaming session if active
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming:
            await self.speech_recognizer.stop_streaming()
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()