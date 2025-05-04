"""
WebSocket handler for Twilio media streams with Deepgram STT integration.
"""
import json
import base64
import asyncio
import logging
import time
import numpy as np
import re
from typing import Dict, Any, Callable, Awaitable, Optional, List
from scipy import signal

from telephony.audio_processor import AudioProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_BUFFER_SIZE

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from speech_to_text.stt_integration import STTIntegration

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
    r'\<.*?noise.*?\>',         # <noise>, etc.
    r'music playing',           # Common transcription
    r'background noise',        # Common transcription
    r'static',                  # Common transcription
]

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams with Deepgram STT integration.
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

        self.last_response = ""
        
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
        
        # Add this flag to fix the attribute error
        self.using_deepgram = False
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Conversation flow management
        self.pause_after_response = 2.0  # Wait 2 seconds after response before processing new input
        self.min_words_for_valid_query = 2  # Minimum words for a valid query
        
        # Create an event to signal when we should stop processing
        self.stop_event = asyncio.Event()
        
        # Add ambient noise tracking for adaptive thresholds
        self.ambient_noise_level = 0.01  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 20
        
        # Determine if we're using GoogleCloudStreamingSTT
        self.using_google_stt = (
            hasattr(pipeline, 'speech_recognizer') and 
            isinstance(pipeline.speech_recognizer, GoogleCloudStreamingSTT)
        )
        
        # Ensure we start with a fresh session state
        self.google_session_active = False
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with Google Cloud STT")
    
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
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples.pop(0)
            
            # Update ambient noise level (with safety floor)
            if self.noise_samples:
                # Use 95th percentile to avoid outliers
                self.ambient_noise_level = max(
                    0.005,  # Minimum threshold
                    np.percentile(self.noise_samples, 95) * 2.0  # Set threshold just above noise
                )
                logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
    
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
            
        # Check if it matches any non-speech patterns
        if self.non_speech_pattern.search(text):
            logger.info(f"Transcription contains non-speech patterns: {text}")
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
        self.silence_start_time = None
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.conversation_active = True
        self.stop_event.clear()
        self.noise_samples = []  # Reset noise samples
        self.deepgram_session_active = False  # Reset Deepgram session state
        
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
        Handle media event with audio data.
        
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
            logger.error(f"Error in audio preprocessing: {e}", exc_info=True)
            return audio_data  # Return original audio if preprocessing fails
    
    async def _process_audio(self, ws) -> None:
        """Process accumulated audio data through the pipeline with Google Cloud STT."""
        try:
            # Convert buffer to PCM with enhanced processing
            try:
                mulaw_bytes = bytes(self.input_buffer)
                
                # Check if buffer is too small to process
                if len(mulaw_bytes) < 1000:
                    logger.debug(f"Audio buffer too small: {len(mulaw_bytes)} bytes")
                    return
                    
                # Convert using the enhanced audio processing
                # Make sure we're calling the instance method correctly
                pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
                
                # Check if we have valid audio data
                if len(pcm_audio) == 0:
                    logger.warning("Empty audio data after conversion")
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    return
                
                # Additional processing to improve recognition
                pcm_audio = self.audio_processor._preprocess_audio(pcm_audio)
                
                # Check for silence before proceeding
                is_silent = self.audio_processor.detect_silence(pcm_audio)
                if is_silent and len(self.input_buffer) < AUDIO_BUFFER_SIZE * 1.5:
                    logger.debug("Detected silence, skipping processing")
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    return
                    
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
            
            # Create a list to collect transcription results
            transcription_results = []
            
            # Define a callback to collect results
            async def transcription_callback(result):
                if hasattr(result, 'is_final') and result.is_final:
                    transcription_results.append(result)
                    logger.debug(f"Received final transcription result: {result.text}")
            
            # Process audio through Google Cloud STT with timeout protection
            try:
                # For Google Cloud, convert to bytes format
                audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
                
                # Ensure Google session is active
                if not self.google_session_active:
                    logger.info("Starting new Google Cloud STT session")
                    try:
                        # Use asyncio timeout to prevent hanging
                        await self.pipeline.speech_recognizer.start_streaming()
                        self.google_session_active = True
                        logger.info("Google Cloud STT session started successfully")
                    except Exception as start_error:
                        logger.error(f"Error starting Google Cloud STT session: {start_error}")
                        self.google_session_active = False
                        return
                
                # Process chunk with Google Cloud
                result = None
                try:
                    result = await self.pipeline.speech_recognizer.process_audio_chunk(
                        audio_chunk=audio_bytes,
                        callback=transcription_callback
                    )
                except Exception as process_error:
                    logger.error(f"Error processing audio chunk: {process_error}")
                    # Continue with available results if any
                
                # Check if we got a final result directly
                if result and hasattr(result, 'is_final') and result.is_final:
                    transcription_results.append(result)
                
                # Get transcription if we have results
                transcription = ""
                if transcription_results:
                    # Use the best result based on confidence
                    best_result = max(transcription_results, key=lambda r: getattr(r, 'confidence', 0))
                    transcription = best_result.text
                elif len(self.input_buffer) > 3 * AUDIO_BUFFER_SIZE:
                    # If we've been collecting audio for a while but no results
                    if self.google_session_active:
                        try:
                            logger.info("No transcription results - stopping and restarting STT session")
                            final_text, _ = await self.pipeline.speech_recognizer.stop_streaming()
                            await asyncio.sleep(0.5)  # Brief pause before restart
                            await self.pipeline.speech_recognizer.start_streaming()
                            self.google_session_active = True
                            transcription = final_text
                            logger.info(f"Final transcription after stop/restart: '{transcription}'")
                        except Exception as restart_error:
                            logger.error(f"Error restarting STT session: {restart_error}")
                            transcription = ""
                            # Reset session state
                            self.google_session_active = False
                
                # Check for barge-in
                barge_in_detected = False
                if self.using_google_stt and hasattr(self.pipeline.speech_recognizer, 'is_barge_in_detected'):
                    barge_in_detected = self.pipeline.speech_recognizer.is_barge_in_detected()
                    if barge_in_detected:
                        logger.info("Barge-in detected, interrupting current response")
                        # Reset barge-in detection
                        self.pipeline.speech_recognizer.reset_barge_in_detection()
            
                # Log before cleanup for debugging
                logger.info(f"RAW TRANSCRIPTION: '{transcription}'")
                
                # Skip processing if empty transcription
                if not transcription:
                    logger.debug("Empty transcription, skipping processing")
                    # Clear buffer and continue
                    self.input_buffer.clear()
                    return
                    
                # Check for echo of system's own speech
                if hasattr(self, 'last_response') and self._is_echo_of_own_speech(transcription):
                    logger.info(f"Detected echo of system speech, ignoring: '{transcription}'")
                    # Clear buffer and skip processing
                    self.input_buffer.clear()
                    return
                
                # Clean up transcription
                transcription = self.cleanup_transcription(transcription)
                logger.info(f"CLEANED TRANSCRIPTION: '{transcription}'")
                
                # Process valid transcription if we have one
                if transcription and len(transcription.split()) >= self.min_words_for_valid_query:
                    # Make sure we don't process the same transcription twice
                    if transcription != self.last_transcription:
                        self.last_transcription = transcription
                        
                        # Generate response through conversation manager
                        try:
                            # Only process if enough time has passed since last response
                            time_since_last_response = time.time() - self.last_response_time
                            if time_since_last_response >= self.pause_after_response:
                                # Process through conversation manager
                                logger.info(f"Processing transcription through conversation manager: '{transcription}'")
                                
                                try:
                                    # Get response from knowledge base
                                    response = await self.pipeline.conversation_manager.handle_user_input(transcription)
                                    response_text = response.get("response", "")
                                    
                                    if response_text:
                                        # Store the response for echo detection
                                        self.last_response = response_text
                                        
                                        # Send response
                                        logger.info(f"Sending response: '{response_text}'")
                                        await self.send_text_response(response_text, ws)
                                        
                                        # Update last response time
                                        self.last_response_time = time.time()
                                        
                                        # Add to conversation history
                                        if hasattr(self.pipeline, 'conversation_manager'):
                                            logger.debug("Adding transcription to conversation history")
                                except Exception as timeout_error:
                                    logger.error(f"Error generating response: {timeout_error}")
                                    await self.send_text_response("I'm sorry, it's taking longer than expected to process your request.", ws)
                            else:
                                logger.debug(f"Skipping processing - too soon after last response ({time_since_last_response:.1f}s < {self.pause_after_response:.1f}s)")
                        except Exception as response_error:
                            logger.error(f"Error generating response: {response_error}")
                            # Send a fallback response
                            await self.send_text_response("I'm sorry, I'm having trouble processing your request.", ws)
                    else:
                        logger.debug(f"Skipping duplicate transcription: '{transcription}'")
                else:
                    logger.debug(f"Transcription not valid for processing: '{transcription}'")
                
                # Clear the input buffer for the next chunk of audio
                self.input_buffer.clear()
                
            except Exception as e:
                logger.error(f"Error during STT processing: {e}")
                # If error, clear part of buffer and continue
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                
                # If we had a Google session error, reset the session
                if self.using_google_stt:
                    try:
                        logger.info("Resetting Google Cloud STT session after error")
                        try:
                            await self.pipeline.speech_recognizer.stop_streaming()
                            await asyncio.sleep(0.5)  # Brief pause
                            await self.pipeline.speech_recognizer.start_streaming()
                            self.google_session_active = True
                        except Exception as timeout_error:
                            logger.error(f"Error during session reset: {timeout_error}")
                            self.google_session_active = False
                    except Exception as session_error:
                        logger.error(f"Error resetting Google Cloud STT session: {session_error}")
                        self.google_session_active = False
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            # Try to continue processing in future calls
    def _is_echo_of_own_speech(self, transcription: str) -> bool:
        """
        Check if transcription is an echo of the system's own speech.
        
        Args:
            transcription: Transcription to check
            
        Returns:
            True if transcription appears to be an echo of own speech
        """
        # Convert to lowercase for comparison
        text = transcription.lower()
        
        # Check against recent responses
        last_response = getattr(self, 'last_response', '').lower()
        
        # If last response is empty, check common greeting phrases
        if not last_response:
            common_greetings = [
                "i'm listening", 
                "how can i help", 
                "welcome to", 
                "how may i assist",
                "i am listening",
                "is there anything"
            ]
            
            # Check if transcription contains any common greeting
            for greeting in common_greetings:
                if greeting in text:
                    logger.info(f"Detected echo of greeting: '{text}'")
                    return True
                    
            return False
        
        # Check if transcription is similar to last response
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, text, last_response).ratio()
        
        # If similarity is high, it's likely an echo
        if similarity > 0.5:
            logger.info(f"Detected echo of own speech (similarity: {similarity:.2f})")
            return True
            
        # Check for partial matches (beginning of response)
        if len(last_response) > 20:
            # Check if transcription is the beginning of the last response
            start_of_response = last_response[:min(len(text) * 2, len(last_response))]
            similarity = SequenceMatcher(None, text, start_of_response).ratio()
            if similarity > 0.5:
                logger.info(f"Detected echo of beginning of response (similarity: {similarity:.2f})")
                return True
        
        return False

    def _detect_silence(self, audio_data: np.ndarray) -> bool:
        """
        Detect if audio is mostly silence using adaptive threshold.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if audio is considered silence
        """
        try:
            # Calculate energy
            energy = np.mean(np.abs(audio_data))
            
            # Use adaptive threshold based on noise floor
            threshold = max(0.01, getattr(self, 'ambient_noise_level', 0.01) * 2.0)
            
            # If energy is below threshold, it's silence
            is_silence = energy < threshold
            
            # Additional check for very quiet audio with adaptive threshold
            if not is_silence and energy < threshold * 2:
                # Check zero-crossing rate (white noise has high ZCR)
                zcr = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
                
                # High ZCR with low energy is likely background noise, not speech
                if zcr > 0.15:
                    logger.debug(f"Detected background noise: energy={energy:.4f}, zcr={zcr:.4f}")
                    is_silence = True
            
            return is_silence
        
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
            # Default to not silence on error
            return False
    
    def _contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced speech detection with better noise filtering.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if audio contains potential speech
        """
        if len(audio_data) < 100:
            return False
            
        # Calculate RMS energy
        energy = np.sqrt(np.mean(np.square(audio_data)))
        
        # Calculate zero-crossing rate (helps distinguish speech from noise)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
        
        # Calculate spectral centroid (speech typically has higher centroids than noise)
        fft_data = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1/16000)
        spectral_centroid = np.sum(freqs * fft_data) / (np.sum(fft_data) + 1e-10)
        
        # Get threshold based on ambient noise
        speech_threshold = max(0.01, self.ambient_noise_level * 2.5)
        
        # Log values for debugging
        logger.debug(f"Audio energy: {energy:.6f} (threshold: {speech_threshold:.6f}), "
                    f"zero crossings: {zero_crossings:.6f}, "
                    f"spectral centroid: {spectral_centroid:.2f}Hz")
        
        # Combined speech detection logic
        is_speech = (energy > speech_threshold) and \
                   (zero_crossings > 0.01) and \
                   (zero_crossings < 0.15) and \
                   (spectral_centroid > 500)  # Speech typically > 500Hz
        
        return is_speech
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """
        Send audio data to Twilio.
        
        Args:
            audio_data: Audio data as bytes
            ws: WebSocket connection
        """
        try:
            # Ensure the audio data is valid
            if not audio_data or len(audio_data) == 0:
                logger.warning("Attempted to send empty audio data")
                return
                
            # Check connection status
            if not self.connected:
                logger.warning("WebSocket connection is closed, cannot send audio")
                return
            
            # Split audio into smaller chunks to avoid timeouts
            chunk_size = 4000  # Smaller chunks (250ms of audio at 8kHz mono)
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            
            logger.debug(f"Splitting {len(audio_data)} bytes into {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
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
                    
                    # Add a small delay between chunks to prevent flooding
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.02)  # 20ms delay between chunks
                    
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
                speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                
                # Convert to mulaw
                mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                
                # Send audio
                await self._send_audio(mulaw_audio, ws)
                logger.info(f"Sent text response: '{text}'")
                
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