"""
WebSocket handler for Twilio media streams with improved speech/noise discrimination and barge-in detection.
"""
import json
import base64
import asyncio
import logging
import time
import numpy as np
import re
from typing import Dict, Any, Callable, Awaitable, Optional, List, Union
from scipy import signal
from collections import deque

from telephony.audio_processor import AudioProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_BUFFER_SIZE
from telephony.config import ENABLE_BARGE_IN, BARGE_IN_THRESHOLD, BARGE_IN_DETECTION_WINDOW

# Import Deepgram STT
from speech_to_text.deepgram_stt import DeepgramStreamingSTT, StreamingTranscriptionResult

logger = logging.getLogger(__name__)

# Enhanced patterns for non-speech annotations
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
    r'\(.*?clap.*?\)',          # (clap), etc.
    r'\(.*?laugh.*?\)',         # (laughing), etc.
    # Additional noise patterns
    r'\[.*?noise.*?\]',         # [noise], etc.
    r'\(.*?background.*?\)',    # (background), etc.
    r'\[.*?music.*?\]',         # [music], etc.
    r'\(.*?static.*?\)',        # (static), etc.
    r'\[.*?unclear.*?\]',       # [unclear], etc.
    r'\(.*?inaudible.*?\)',     # (inaudible), etc.
    r'<.*?noise.*?>',         # <noise>, etc.
    r'music playing',           # Common transcription
    r'background noise',        # Common transcription
    r'static',                  # Common transcription
]

# Speech state enum for state machine
class SpeechState:
    """Speech state for detection state machine"""
    SILENCE = 0
    POTENTIAL_SPEECH = 1
    CONFIRMED_SPEECH = 2
    SPEECH_ENDED = 3

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams with enhanced speech/noise discrimination.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        self.audio_processor = AudioProcessor()
        
        # Audio buffers
        self.input_buffer = bytearray()
        self.output_buffer = bytearray()
        
        # State tracking
        self.is_speaking = False
        self.silence_start_time = None
        self.is_processing = False
        self.conversation_active = True
        self.sequence_number = 0  # For Twilio media sequence tracking
        
        # Connection state tracking
        self.connected = False
        self.connection_active = asyncio.Event()
        self.connection_active.clear()
        
        # Transcription tracker to avoid duplicate processing
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.processing_lock = asyncio.Lock()
        self.keep_alive_task = None
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Conversation flow management
        self.pause_after_response = 2.0  # Wait 2 seconds after response before processing new input
        self.min_words_for_valid_query = 2  # Minimum words for a valid query
        
        # Create an event to signal when we should stop processing
        self.stop_event = asyncio.Event()
        
        # Enhanced adaptive noise tracking for better speech/noise discrimination
        self.ambient_noise_level = 0.01  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 30  # Increased from 20 for better statistics
        self.noise_floor_multiplier = 2.5  # Multiplier for noise floor to set low threshold
        
        # Speech frequency band energies for better detection
        self.speech_band_energies = deque(maxlen=20)
        
        # Determine if we're using Deepgram STT
        self.using_deepgram = (
            hasattr(pipeline, 'speech_recognizer') and 
            isinstance(pipeline.speech_recognizer, DeepgramStreamingSTT)
        )
        
        # Ensure we start with a fresh deepgram session state
        self.deepgram_session_active = False
        
        # Enhanced barge-in detection with state machine
        self.is_agent_speaking = False
        self.can_be_interrupted = ENABLE_BARGE_IN
        self.barge_in_detected = False
        self.speech_cancellation_event = asyncio.Event()
        self.speech_cancellation_event.clear()
        
        # Double-threshold barge-in detection with state machine
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0  # Count of consecutive potential speech frames
        self.confirmed_speech_frames = 0  # Count of consecutive confirmed speech frames
        self.potential_speech_threshold = self.ambient_noise_level * 2.0  # Lower threshold
        self.confirmed_speech_threshold = self.ambient_noise_level * 3.5  # Higher threshold
        self.min_speech_frames_for_barge_in = 5  # Minimum frames to consider interruption (was 3)
        
        # Maintain enhanced audio buffer for interruption detection
        self.recent_audio_buffer = deque(maxlen=15)  # Increased from 8 to 15 chunks for better detection
        self.barge_in_detection_window = BARGE_IN_DETECTION_WINDOW  # ms
        self.barge_in_threshold = BARGE_IN_THRESHOLD
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with {'Deepgram' if self.using_deepgram else 'Whisper'} STT")
        logger.info(f"Barge-in functionality {'enabled' if self.can_be_interrupted else 'disabled'}")
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
        """
        Enhanced adaptive noise level tracking with better statistical approach.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Calculate energy of the audio
        energy = np.mean(np.abs(audio_data))
        
        # If audio is silence (very low energy), use it to update noise floor
        if energy < 0.02:  # Very quiet audio
            self.noise_samples.append(energy)
            # Keep only recent samples
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples.pop(0)
            
            # Update ambient noise level (with safety floor)
            if self.noise_samples:
                # Use 90th percentile to avoid outliers
                self.ambient_noise_level = max(
                    0.005,  # Minimum threshold
                    np.percentile(self.noise_samples, 90) * 2.0  # Set threshold just above noise
                )
                logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
                
                # Update speech detection thresholds
                self.potential_speech_threshold = self.ambient_noise_level * self.noise_floor_multiplier
                self.confirmed_speech_threshold = self.ambient_noise_level * 3.5
    
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
        
        # Be more lenient with question marks and punctuation in confidence calculation
        confidence_estimate = 1.0
        if ("[" in text or "(" in text or "<" in text) and not "?" in text:
            confidence_estimate = 0.7  # Only reduce for annotation markers, not question marks
            logger.info(f"Reduced confidence due to uncertainty markers: {text}")
        
        if confidence_estimate < 0.65:
            logger.info(f"Transcription confidence too low: {confidence_estimate}")
            return False
        
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_valid_query:
            logger.info(f"Transcription too short: {word_count} words")
            return False
        
        return True
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        if not message:
            logger.warning("Received empty message")
            return
            
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            logger.debug(f"Received WebSocket event: {event_type}")
            
            # Handle different event types
            if event_type == 'connected':
                await self._handle_connected(data, ws)
            elif event_type == 'start':
                await self._handle_start(data, ws)
            elif event_type == 'media':
                await self._handle_media(data, ws)
            elif event_type == 'stop':
                await self._handle_stop(data)
            elif event_type == 'mark':
                await self._handle_mark(data)
            else:
                logger.warning(f"Unknown event type: {event_type}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _handle_connected(self, data: Dict[str, Any], ws) -> None:
        """
        Handle connected event.
        
        Args:
            data: Connected event data
            ws: WebSocket connection
        """
        logger.info(f"WebSocket connected for call {self.call_sid}")
        logger.info(f"Connected data: {data}")
        
        # Set connection state
        self.connected = True
        self.connection_active.set()
        
        # Start keep-alive task
        if not self.keep_alive_task:
            self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """
        Handle stream start event.
        
        Args:
            data: Start event data
            ws: WebSocket connection
        """
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        logger.info(f"Start data: {data}")
        
        # Reset state for new stream
        self.input_buffer.clear()
        self.output_buffer.clear()
        self.is_speaking = False
        self.is_processing = False
        self.is_agent_speaking = False
        self.barge_in_detected = False
        self.speech_cancellation_event.clear()
        self.silence_start_time = None
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.conversation_active = True
        self.stop_event.clear()
        self.noise_samples = []  # Reset noise samples
        self.deepgram_session_active = False  # Reset Deepgram session state
        self.recent_audio_buffer.clear()  # Reset audio buffer for barge-in detection
        self.speech_state = SpeechState.SILENCE  # Reset speech state machine
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        self.speech_band_energies.clear()  # Clear speech band energies
        
        # Send a welcome message
        await self.send_text_response("I'm listening. How can I help you today?", ws)
        
        # Initialize Deepgram streaming session if using Deepgram
        if self.using_deepgram:
            try:
                await self.pipeline.speech_recognizer.start_streaming()
                self.deepgram_session_active = True
                logger.info("Started Deepgram streaming session")
            except Exception as e:
                logger.error(f"Error starting Deepgram streaming session: {e}")
                self.deepgram_session_active = False
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """
        Handle media event with audio data and enhanced speech/noise discrimination.
        
        Args:
            data: Media event data
            ws: WebSocket connection
        """
        if not self.conversation_active:
            logger.debug("Conversation not active, ignoring media")
            return
            
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning("Received media event with no payload")
            return
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            
            # Add to input buffer
            self.input_buffer.extend(audio_data)
            
            # Check for barge-in if agent is speaking
            if self.is_agent_speaking and self.can_be_interrupted:
                await self._check_for_barge_in(audio_data, ws)
            
            # Limit buffer size to prevent memory issues
            if len(self.input_buffer) > MAX_BUFFER_SIZE:
                # Keep the most recent portion
                excess = len(self.input_buffer) - MAX_BUFFER_SIZE
                self.input_buffer = self.input_buffer[excess:]
                logger.debug(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
            
            # Check if we should process based on time since last response
            time_since_last_response = time.time() - self.last_response_time
            if time_since_last_response < self.pause_after_response:
                # Still in pause period after last response, wait before processing new input
                logger.debug(f"In pause period after response ({time_since_last_response:.1f}s < {self.pause_after_response:.1f}s)")
                return
            
            # Process buffer when it's large enough and not already processing
            if len(self.input_buffer) >= AUDIO_BUFFER_SIZE and not self.is_processing:
                async with self.processing_lock:
                    if not self.is_processing:  # Double-check within lock
                        self.is_processing = True
                        try:
                            logger.info(f"Processing audio buffer of size: {len(self.input_buffer)} bytes")
                            await self._process_audio(ws)
                        finally:
                            self.is_processing = False
            
        except Exception as e:
            logger.error(f"Error processing media payload: {e}", exc_info=True)
    
    async def _check_for_barge_in(self, audio_data: bytes, ws) -> None:
        """
        Enhanced barge-in detection with double-threshold approach and state machine.
        
        Args:
            audio_data: Raw audio data
            ws: WebSocket connection
        """
        try:
            # Convert μ-law to PCM
            pcm_audio = self.audio_processor.mulaw_to_pcm(audio_data)
            
            # Add to recent audio buffer for detection window
            self.recent_audio_buffer.append(pcm_audio)
            
            # Only attempt barge-in detection if we have enough audio
            if len(self.recent_audio_buffer) >= 5:  # Need at least 5 chunks (100ms) of audio
                # Combine the most recent audio frames
                combined_audio = np.concatenate(list(self.recent_audio_buffer))
                
                # Process audio with enhanced preprocessing
                enhanced_audio = self._preprocess_audio(combined_audio)
                
                # Calculate energy metrics
                rms_energy = np.sqrt(np.mean(np.square(enhanced_audio)))
                
                # Calculate speech band energy
                band_energies = self._calculate_speech_band_energy(enhanced_audio)
                speech_ratio = band_energies["speech_ratio"]
                
                # State machine for barge-in detection (reduces false positives)
                if self.speech_state == SpeechState.SILENCE:
                    # Check if potential speech detected (lower threshold)
                    if rms_energy > self.potential_speech_threshold and speech_ratio > 0.5:
                        self.speech_state = SpeechState.POTENTIAL_SPEECH
                        self.potential_speech_frames = 1
                        logger.debug(f"Potential speech detected: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                
                elif self.speech_state == SpeechState.POTENTIAL_SPEECH:
                    # Check if we still have potential speech
                    if rms_energy > self.potential_speech_threshold and speech_ratio > 0.5:
                        self.potential_speech_frames += 1
                        # Check if we have enough frames to move to confirmed speech
                        if self.potential_speech_frames >= 3:  # Require 3 consecutive frames
                            # Check against higher threshold for confirmation
                            if rms_energy > self.confirmed_speech_threshold and speech_ratio > 0.6:
                                self.speech_state = SpeechState.CONFIRMED_SPEECH
                                self.confirmed_speech_frames = 1
                                logger.info(f"Speech confirmed: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                    else:
                        # Drop back to silence state
                        self.speech_state = SpeechState.SILENCE
                        self.potential_speech_frames = 0
                
                elif self.speech_state == SpeechState.CONFIRMED_SPEECH:
                    # Check if we still have confirmed speech
                    if rms_energy > self.confirmed_speech_threshold:
                        self.confirmed_speech_frames += 1
                        
                        # Check if we have enough frames to trigger barge-in
                        if self.confirmed_speech_frames >= self.min_speech_frames_for_barge_in:
                            logger.info(f"Barge-in detected! Sustained speech for {self.confirmed_speech_frames} frames")
                            logger.info(f"Speech energy: {rms_energy:.4f}, speech_ratio: {speech_ratio:.2f}")
                            
                            # Set barge-in flag and trigger cancellation
                            self.barge_in_detected = True
                            self.speech_cancellation_event.set()
                            
                            # This adds a delay to avoid rapid back-and-forth interruptions
                            await asyncio.sleep(0.3)  # 300ms debounce
                            
                            # Send silence and process interruption
                            try:
                                silence_data = b'\x00' * 160
                                silent_mulaw = self.audio_processor.pcm_to_mulaw(silence_data)
                                await self._send_audio(silent_mulaw, ws, is_interrupt=True)
                                
                                # Reset agent speaking state
                                self.is_agent_speaking = False
                                
                                # Reset speech state machine
                                self.speech_state = SpeechState.SILENCE
                                self.potential_speech_frames = 0
                                self.confirmed_speech_frames = 0
                                
                                # Clear recent audio buffer
                                self.recent_audio_buffer.clear()
                                
                                # Only process interruption if we have enough audio
                                if len(self.input_buffer) > 2000:  # Require more audio (2000 bytes)
                                    logger.info("Processing interruption immediately")
                                    await self._process_audio(ws)
                            except Exception as barge_in_error:
                                logger.error(f"Error handling barge-in: {barge_in_error}")
                    else:
                        # Drop back to potential speech or silence
                        if rms_energy > self.potential_speech_threshold:
                            self.speech_state = SpeechState.POTENTIAL_SPEECH
                            self.potential_speech_frames = 1
                            self.confirmed_speech_frames = 0
                        else:
                            self.speech_state = SpeechState.SILENCE
                            self.potential_speech_frames = 0
                            self.confirmed_speech_frames = 0

        except Exception as e:
            logger.error(f"Error checking for barge-in: {e}", exc_info=True)
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """
        Handle stream stop event.
        
        Args:
            data: Stop event data
        """
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.conversation_active = False
        self.connected = False
        self.connection_active.clear()
        self.stop_event.set()
        self.speech_cancellation_event.set()  # Cancel any ongoing speech
        
        # Cancel keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Close Deepgram streaming session if using Deepgram
        if self.using_deepgram and self.deepgram_session_active:
            try:
                await self.pipeline.speech_recognizer.stop_streaming()
                logger.info("Stopped Deepgram streaming session")
                self.deepgram_session_active = False
            except Exception as e:
                logger.error(f"Error stopping Deepgram streaming session: {e}")
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """
        Handle mark event for audio playback tracking.
        
        Args:
            data: Mark event data
        """
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhanced audio preprocessing with spectral gating and speech emphasis.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Preprocessed audio data
        """
        try:
            # 1. Apply high-pass filter to remove low-frequency noise (below 100Hz)
            b, a = signal.butter(6, 100/(16000/2), 'highpass')
            audio_data = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply band-pass filter for telephony frequency range (300-3400 Hz)
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            audio_data = signal.filtfilt(b, a, audio_data)
            
            # 3. Apply spectral gating (simplified implementation)
            if len(audio_data) >= 512:  # Enough samples for FFT
                # Compute STFT
                f, t, Zxx = signal.stft(audio_data, fs=16000, nperseg=512, noverlap=384)
                
                # Estimate noise spectrum from lowest 10% of magnitudes
                magnitude = np.abs(Zxx)
                noise_threshold = np.percentile(magnitude, 10, axis=1, keepdims=True)
                
                # Apply soft spectral gating
                gain = 1.0 - np.exp(-magnitude**2 / (noise_threshold**2 + 1e-10))
                Zxx_cleaned = Zxx * gain
                
                # Inverse STFT
                _, audio_data = signal.istft(Zxx_cleaned, fs=16000, nperseg=512, noverlap=384)
                
                # Ensure same length as original
                if len(audio_data) > len(audio_data):
                    audio_data = audio_data[:len(audio_data)]
                elif len(audio_data) < len(audio_data):
                    padding = np.zeros(len(audio_data) - len(audio_data))
                    audio_data = np.concatenate([audio_data, padding])
            
            # 4. Apply noise gate with adaptive threshold
            threshold = max(0.015, self.ambient_noise_level)
            audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
            
            # 5. Apply pre-emphasis to boost high frequencies (speech clarity)
            audio_data = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])
            
            # 6. Speech onset emphasis - boost sudden changes in energy
            if len(audio_data) > 320:  # At least 20ms
                # Calculate energy envelope
                frame_size = 160  # 10ms
                energy_envelope = np.array([
                    np.mean(np.square(audio_data[i:i+frame_size]))
                    for i in range(0, len(audio_data)-frame_size, frame_size)
                ])
                
                # Compute derivative of energy
                energy_derivative = np.diff(energy_envelope, prepend=energy_envelope[0])
                
                # Find regions of rising energy (speech onset)
                rising_energy = energy_derivative > 0
                
                # Expand to full audio length
                rising_energy_full = np.repeat(rising_energy, frame_size)
                if len(rising_energy_full) < len(audio_data):
                    rising_energy_full = np.pad(
                        rising_energy_full,
                        (0, len(audio_data) - len(rising_energy_full)),
                        'edge'
                    )
                
                # Apply gentle boost to regions of rising energy
                boost_factor = np.ones_like(audio_data)
                boost_factor[rising_energy_full] = 1.2  # 20% boost for speech onsets
                audio_data = audio_data * boost_factor
            
            # 7. Normalize audio to have consistent volume
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data * (0.9 / max_val)
            
            # Update ambient noise level (for future processing)
            self._update_ambient_noise_level(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}", exc_info=True)
            return audio_data  # Return original if processing fails
    
    async def _process_audio(self, ws) -> None:
        """
        Process accumulated audio data through the pipeline with enhanced speech processing.
        
        Args:
            ws: WebSocket connection
        """
        try:
            # Convert buffer to PCM with enhanced processing
            try:
                mulaw_bytes = bytes(self.input_buffer)
                
                # Convert using the enhanced audio processing
                pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
                
                # Additional processing to improve recognition
                pcm_audio = self._preprocess_audio(pcm_audio)
                
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                # Clear part of buffer and try again next time
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
                
            # Add some checks for audio quality
            if len(pcm_audio) < 1000:  # Very small audio chunk
                logger.warning(f"Audio chunk too small: {len(pcm_audio)} samples")
                return
            
            # Check if audio contains actual speech using our enhanced detection
            contains_speech = self._contains_speech(pcm_audio)
            if not contains_speech:
                logger.info("No speech detected in audio buffer, skipping processing")
                # Reduce buffer size but keep some context
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
            
            # Create a list to collect transcription results
            transcription_results = []
            
            # Define a callback to collect results
            async def transcription_callback(result):
                if self.using_deepgram:
                    # For Deepgram, we care about final results
                    if result.is_final:
                        transcription_results.append(result)
                        logger.debug(f"Received final Deepgram result: {result.text}")
                else:
                    # For Whisper, all results are collected
                    transcription_results.append(result)
                    logger.debug(f"Received Whisper result: {result.text}")
            
            # Process audio through the appropriate STT system
            try:
                if self.using_deepgram:
                    # For Deepgram, convert to bytes format
                    audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
                    
                    # Make sure the Deepgram streaming session is active
                    if not self.deepgram_session_active:
                        logger.info("Starting new Deepgram streaming session")
                        await self.pipeline.speech_recognizer.start_streaming()
                        self.deepgram_session_active = True
                    
                    # Process chunk with Deepgram
                    result = await self.pipeline.speech_recognizer.process_audio_chunk(
                        audio_chunk=audio_bytes,
                        callback=transcription_callback
                    )
                    
                    # Check if we got a final result directly
                    if result and result.is_final:
                        transcription_results.append(result)
                    
                    # Get transcription if we have results
                    if transcription_results:
                        # Use the best result based on confidence
                        best_result = max(transcription_results, key=lambda r: r.confidence)
                        transcription = best_result.text
                    else:
                        # If no results, try stopping and restarting the session to get final results
                        if self.deepgram_session_active:
                            await self.pipeline.speech_recognizer.stop_streaming()
                            await self.pipeline.speech_recognizer.start_streaming()
                            self.deepgram_session_active = True
                        
                        # No transcription
                        transcription = ""
                else:
                    # For Whisper, use the original approach
                    # Ensure we're starting fresh
                    if hasattr(self.pipeline.speech_recognizer, 'is_streaming') and self.pipeline.speech_recognizer.is_streaming:
                        await self.pipeline.speech_recognizer.stop_streaming()
                    
                    # Start a new streaming session
                    self.pipeline.speech_recognizer.start_streaming()
                    
                    # Process audio chunk
                    await self.pipeline.speech_recognizer.process_audio_chunk(
                        audio_chunk=pcm_audio,
                        callback=transcription_callback
                    )
                    
                    # Get final transcription
                    transcription, _ = await self.pipeline.speech_recognizer.stop_streaming()
                
                # Log before cleanup for debugging
                logger.info(f"RAW TRANSCRIPTION: '{transcription}'")
                
                # Clean up transcription
                transcription = self.cleanup_transcription(transcription)
                logger.info(f"CLEANED TRANSCRIPTION: '{transcription}'")
                
                # Only process if it's a valid transcription
                if transcription and self.is_valid_transcription(transcription):
                    logger.info(f"Complete transcription: {transcription}")
                    
                    # Now clear the input buffer since we have a valid transcription
                    self.input_buffer.clear()
                    
                    # Don't process duplicate transcriptions
                    if transcription == self.last_transcription:
                        logger.info("Duplicate transcription, not processing again")
                        return
                    
                    # Process through knowledge base
                    try:
                        if hasattr(self.pipeline, 'query_engine'):
                            query_result = await self.pipeline.query_engine.query(transcription)
                            response = query_result.get("response", "")
                            
                            logger.info(f"Generated response: {response}")
                            
                            # Convert to speech
                            if response and hasattr(self.pipeline, 'tts_integration'):
                                try:
                                    # Set agent speaking state before sending audio
                                    self.is_agent_speaking = True
                                    self.barge_in_detected = False
                                    self.speech_cancellation_event.clear()
                                    
                                    # Reset speech state machine
                                    self.speech_state = SpeechState.SILENCE
                                    self.potential_speech_frames = 0
                                    self.confirmed_speech_frames = 0
                                    
                                    speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                    
                                    # Convert to mulaw for Twilio
                                    mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                                    
                                    # Send back to Twilio
                                    logger.info(f"Sending audio response ({len(mulaw_audio)} bytes)")
                                    await self._send_audio(mulaw_audio, ws)
                                    
                                except Exception as tts_error:
                                    logger.error(f"TTS Error: {tts_error}. Sending fallback response.", exc_info=True)
                                    
                                    # Try to send a text-based fallback response
                                    fallback_message = "I'm having trouble generating speech at the moment. Let me try again in a moment."
                                    
                                    try:
                                        # Try to use a simple SSML template that works with all voices
                                        simple_audio = await self.pipeline.tts_integration.tts_client.synthesize(
                                            "<speak>" + fallback_message + "</speak>", 
                                            is_ssml=True
                                        )
                                        fallback_mulaw = self.audio_processor.pcm_to_mulaw(simple_audio)
                                        await self._send_audio(fallback_mulaw, ws)
                                    except Exception as fallback_error:
                                        logger.error(f"Failed to generate fallback response: {fallback_error}")
                                        
                                        # Use a pre-recorded or simple beep as last resort
                                        silence_duration = 0.5  # seconds
                                        silence_size = int(8000 * silence_duration)
                                        fallback_audio = b'\x7f' * silence_size  # Simple beep
                                        await self._send_audio(fallback_audio, ws)
                                
                                # Reset agent speaking state
                                self.is_agent_speaking = False
                                
                                # Update state
                                self.last_transcription = transcription
                                self.last_response_time = time.time()
                    except Exception as e:
                        logger.error(f"Error processing through knowledge base: {e}", exc_info=True)
                        
                        # Try to send a fallback response
                        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
                        try:
                            # Use simple SSML template
                            fallback_audio = await self.pipeline.tts_integration.tts_client.synthesize(
                                "<speak>" + fallback_message + "</speak>",
                                is_ssml=True
                            )
                            mulaw_fallback = self.audio_processor.pcm_to_mulaw(fallback_audio)
                            await self._send_audio(mulaw_fallback, ws)
                            self.last_response_time = time.time()
                        except Exception as e2:
                            logger.error(f"Failed to send fallback response: {e2}")
                else:
                    # If no valid transcription, reduce buffer size but keep some for context
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    logger.debug(f"No valid transcription, reduced buffer to {len(self.input_buffer)} bytes")
            
            except Exception as e:
                logger.error(f"Error during STT processing: {e}", exc_info=True)
                # If error, clear part of buffer and continue
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                
                # If we had a Deepgram session error, reset the session
                if self.using_deepgram:
                    try:
                        logger.info("Resetting Deepgram session after error")
                        await self.pipeline.speech_recognizer.stop_streaming()
                        await self.pipeline.speech_recognizer.start_streaming()
                        self.deepgram_session_active = True
                    except Exception as session_error:
                        logger.error(f"Error resetting Deepgram session: {session_error}")
                        self.deepgram_session_active = False
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
    
    def _contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced speech detection with improved speech/noise discrimination.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if audio contains speech
        """
        if len(audio_data) < 500:  # Need enough samples
            return False
            
        try:
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(np.square(audio_data)))
            
            # Calculate zero-crossing rate (helps distinguish speech from noise)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
            
            # Calculate frequency band energies
            band_energies = self._calculate_speech_band_energy(audio_data)
            speech_ratio = band_energies["speech_ratio"]
            
            # Calculate spectral flux - measure of how quickly spectrum changes
            # (speech has rapid changes, background noise is more constant)
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
            energy_threshold = max(0.015, self.ambient_noise_level * 3.0)
            
            # Calculate the average speech band ratio from history
            avg_speech_ratio = 0.4  # Default
            if self.speech_band_energies:
                avg_speech_ratio = np.mean(self.speech_band_energies)
            
            # Log detailed values
            logger.debug(f"Speech detection - "
                         f"energy: {rms_energy:.4f} (threshold: {energy_threshold:.4f}), "
                         f"ZCR: {zero_crossings:.4f}, speech_ratio: {speech_ratio:.2f}, "
                         f"avg_speech_ratio: {avg_speech_ratio:.2f}, flux: {spectral_flux:.4f}")
            
            # Combined speech detection with weighting towards speech frequencies
            is_speech = (
                # Must have sufficient energy
                (rms_energy > energy_threshold) and
                
                # Zero crossing rate criteria (avoids pure tones and white noise)
                (zero_crossings > 0.01 and zero_crossings < 0.15) and
                
                # Must have significant energy in speech band or high spectral flux
                ((speech_ratio > 0.5 and speech_ratio > avg_speech_ratio * 0.8) or
                 spectral_flux > 0.2)
            )
            
            if is_speech:
                logger.debug("Speech detected in audio")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            # Fall back to simple energy threshold
            energy = np.mean(np.abs(audio_data))
            return energy > max(0.015, self.ambient_noise_level * 3.0)
    
    async def _send_audio(self, audio_data: bytes, ws, is_interrupt: bool = False) -> None:
        """
        Send audio data to Twilio with barge-in support.
        
        Args:
            audio_data: Audio data as bytes
            ws: WebSocket connection
            is_interrupt: Whether this is a barge-in interruption
        """
        try:
            # Ensure the audio data is valid
            if not audio_data or len(audio_data) == 0:
                logger.warning("Attempted to send empty audio data")
                return
                
            # Skip very small chunks unless they're for interruption
            if len(audio_data) < 320 and not is_interrupt:
                logger.debug(f"Skipping very small audio chunk: {len(audio_data)} bytes")
                return
                
            # Check connection status
            if not self.connected:
                logger.warning("WebSocket connection is closed, cannot send audio")
                return
            
            # Get audio info for debugging
            if not is_interrupt:
                audio_info = self.audio_processor.get_audio_info(audio_data)
                logger.debug(f"Sending audio: format={audio_info.get('format', 'unknown')}, size={len(audio_data)} bytes")
            
            # Split audio into smaller chunks to avoid timeouts
            chunk_size = 320  # 20ms of 8kHz μ-law mono audio
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            
            logger.debug(f"Splitting {len(audio_data)} bytes into {len(chunks)} chunks")
            
            # Set agent speaking flag if this isn't an interrupt signal
            if not is_interrupt:
                self.is_agent_speaking = True
            
            for i, chunk in enumerate(chunks):
                # Check if we should cancel speech (barge-in detected)
                if self.barge_in_detected or self.speech_cancellation_event.is_set():
                    logger.info(f"Cancelling speech output at chunk {i}/{len(chunks)} due to barge-in")
                    # Only send a few more chunks to avoid abrupt cutoff
                    if i > 5:  # We've sent enough to avoid an abrupt stop
                        break
                    # Otherwise, continue with a few more chunks for a smoother transition
                
                try:
                    # Encode audio to base64
                    audio_base64 = base64.b64encode(chunk).decode('utf-8')
                    
                    # Create media message
                    message = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": audio_base64
                        }
                    }
                    
                    # Send message
                    ws.send(json.dumps(message))
                    
                    # Add a small delay between chunks to prevent flooding and allow
                    # for barge-in detection during speech
                    if i < len(chunks) - 1 and i % 5 == 0:  # Every 5 chunks (~100ms)
                        await asyncio.sleep(0.01)  # 10ms delay 
                    
                except Exception as e:
                    if "Connection closed" in str(e):
                        logger.warning(f"WebSocket connection closed while sending chunk {i+1}/{len(chunks)}")
                        self.connected = False
                        self.connection_active.clear()
                        return
                    else:
                        logger.error(f"Error sending audio chunk {i+1}/{len(chunks)}: {e}")
                        return
            
            logger.debug(f"Sent {len(chunks)} audio chunks ({len(audio_data)} bytes total)")
            
            # Reset speaking flag after sending all chunks
            if not is_interrupt:
                self.is_agent_speaking = False
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=True)
            if "Connection closed" in str(e):
                self.connected = False
                self.connection_active.clear()
    
    async def send_text_response(self, text: str, ws) -> None:
        """
        Send a text response by converting to speech first.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        try:
            # Convert text to speech
            if hasattr(self.pipeline, 'tts_integration'):
                # Set agent speaking state
                self.is_agent_speaking = True
                self.barge_in_detected = False
                self.speech_cancellation_event.clear()
                
                # Reset speech state machine
                self.speech_state = SpeechState.SILENCE
                self.potential_speech_frames = 0
                self.confirmed_speech_frames = 0
                
                speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                
                # Convert to mulaw
                mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                
                # Send audio
                await self._send_audio(mulaw_audio, ws)
                logger.info(f"Sent text response: '{text}'")
                
                # Reset agent speaking state
                self.is_agent_speaking = False
                
                # Update last response time to add pause
                self.last_response_time = time.time()
            else:
                logger.error("TTS integration not available")
        except Exception as e:
            logger.error(f"Error sending text response: {e}", exc_info=True)
    
    async def _keep_alive_loop(self, ws) -> None:
        """
        Send periodic keep-alive messages to maintain the WebSocket connection.
        """
        try:
            while self.conversation_active:
                await asyncio.sleep(10)  # Send every 10 seconds
                
                # Only send if we have a valid stream
                if not self.stream_sid or not self.connected:
                    continue
                    
                try:
                    message = {
                        "event": "ping",
                        "streamSid": self.stream_sid
                    }
                    ws.send(json.dumps(message))
                    logger.debug("Sent keep-alive ping")
                except Exception as e:
                    logger.error(f"Error sending keep-alive: {e}")
                    if "Connection closed" in str(e):
                        self.connected = False
                        self.connection_active.clear()
                        self.conversation_active = False
                        break
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")