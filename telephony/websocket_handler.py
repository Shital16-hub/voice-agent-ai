"""
Enhanced WebSocket handler for Twilio media streams with improved speech/noise
discrimination and barge-in detection using the new AudioPreprocessor.
"""
import json
import base64
import asyncio
import logging
import time
import numpy as np
import re
from typing import Dict, Any, Callable, Awaitable, Optional, List, Union, Tuple
from collections import deque

from telephony.config import (
    CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, 
    MAX_BUFFER_SIZE, ENABLE_BARGE_IN, BARGE_IN_THRESHOLD, BARGE_IN_DETECTION_WINDOW
)

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
    r'<.*?noise.*?>',           # <noise>, etc.
    r'music playing',           # Common transcription
    r'background noise',        # Common transcription
    r'static',                  # Common transcription
    r'\*.*?\*',                 # *any text* (scoff, etc.)
    r'^\(.*?\)$',               # Entire text is (in parentheses)
    r'clicking',                # keyboard/tongue clicking
    r'clacking',                # keyboard clacking
    r'tongue',                  # tongue clicks
    r'keyboard',                # keyboard noises
]

# Speech state enum for detection
class SpeechState:
    """Speech state for detection state machine"""
    SILENCE = 0
    POTENTIAL_SPEECH = 1
    CONFIRMED_SPEECH = 2
    SPEECH_ENDED = 3

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams with enhanced speech/noise discrimination.
    Implements advanced barge-in detection directly in this class.
    """
    
    # Add these constants for command recognition
    STOP_COMMANDS = ["stop", "okay", "ok", "enough", "halt", "pause", "quit", "exit", "bye", "end"]
    SIMPLE_COMMANDS = {
        "stop": "stop",
        "okay": "stop",
        "ok": "stop",
        "enough": "stop",
        "halt": "stop",
        "pause": "stop",
        "quit": "stop",
        "exit": "stop",
        "bye": "stop",
        "end": "stop",
        "continue": "continue",
        "go on": "continue",
        "proceed": "continue",
        "yes": "continue",
        "help": "help",
        "repeat": "repeat",
        "again": "repeat",
        "what": "repeat"
    }
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler with Google Cloud STT integration.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Get STT integration from pipeline
        self.stt_integration = pipeline.speech_recognizer.stt_integration if hasattr(pipeline.speech_recognizer, 'stt_integration') else None
        
        # Ensure we have STT integration
        if not self.stt_integration:
            from integration.stt_integration import STTIntegration
            self.stt_integration = STTIntegration(pipeline.speech_recognizer)
        
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
        self.pause_after_response = 3.0  # Wait 3 seconds after response
        self.min_words_for_valid_query = 3  # Minimum 3 words for a valid query
        
        # Create an event to signal when we should stop processing
        self.stop_event = asyncio.Event()
        
        # Barge-in handling
        self.speech_cancellation_event = asyncio.Event()
        self.speech_cancellation_event.clear()
        
        # Add startup delay before processing audio
        self.startup_time = time.time()
        self.startup_delay_sec = 2.0  # 2 second delay before processing audio
        
        # Track chunks processed since last processing 
        self.chunks_since_last_process = 0
        
        # Implement advanced barge-in detection directly
        # Enhanced adaptive noise tracking
        self.noise_samples = []
        self.max_samples = 50
        self.ambient_noise_level = 0.015
        self.min_noise_floor = 0.008
        
        # Enhanced energy thresholds with hysteresis
        self.low_threshold = self.ambient_noise_level * 2.0
        self.high_threshold = self.ambient_noise_level * 4.5
        
        # Speech frequency band energy tracking
        self.speech_band_energies = deque(maxlen=30)
        
        # Speech detection state machine
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        self.silence_frames = 0
        
        # Maintain enhanced audio buffer for improved detection
        self.recent_audio_buffer = deque(maxlen=20)
        
        # Barge-in state tracking
        self.agent_speaking = False
        self.agent_speaking_start_time = 0.0
        self.barge_in_detected = False
        self.last_barge_in_time = 0.0
        
        # Set barge-in parameters
        self.barge_in_threshold = BARGE_IN_THRESHOLD
        self.barge_in_min_speech_frames = 10
        self.barge_in_cooldown_ms = 2000
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with Google Cloud STT")
    
    def _is_simple_command(self, text: str) -> Tuple[bool, str]:
        """
        Check if text is a simple command like "stop", "okay", etc.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_command, command_type)
        """
        if not text:
            return False, ""
            
        # Clean and lowercase the text
        clean_text = text.lower().strip().rstrip('.!?')
        
        # Direct matches
        if clean_text in self.SIMPLE_COMMANDS:
            return True, self.SIMPLE_COMMANDS[clean_text]
        
        # Check for commands embedded in longer phrases
        for command, action in self.SIMPLE_COMMANDS.items():
            if f" {command} " in f" {clean_text} ":
                return True, action
                
        return False, ""
    
    def _handle_command(self, command_type: str, ws) -> bool:
        """
        Handle a recognized command.
        
        Args:
            command_type: Type of command
            ws: WebSocket connection
            
        Returns:
            True if command was handled
        """
        if command_type == "stop":
            # Interrupt any ongoing speech
            logger.info("Stop command detected, interrupting speech")
            self.speech_cancellation_event.set()
            
            # Send confirmation to user
            asyncio.create_task(self.send_text_response("Okay, I'll stop.", ws))
            
            # Reset buffer
            self.input_buffer.clear()
            return True
            
        elif command_type == "continue":
            # Acknowledge continue command
            logger.info("Continue command detected")
            asyncio.create_task(self.send_text_response("I'll continue.", ws))
            return True
            
        elif command_type == "help":
            # Provide help
            logger.info("Help command detected")
            help_text = "You can say things like 'stop', 'repeat', or ask me questions about our services."
            asyncio.create_task(self.send_text_response(help_text, ws))
            return True
            
        elif command_type == "repeat":
            # Repeat last response
            logger.info("Repeat command detected")
            if hasattr(self, 'last_transcription') and hasattr(self, 'last_response_time'):
                if self.last_transcription:
                    # Reprocess the last query
                    asyncio.create_task(self._process_last_query(ws))
                    return True
            
            # No last response to repeat
            asyncio.create_task(self.send_text_response("I don't have anything to repeat yet.", ws))
            return True
            
        return False
        
    async def _process_last_query(self, ws):
        """Reprocess the last valid query."""
        try:
            if hasattr(self, 'last_transcription') and self.last_transcription:
                # Process through knowledge base
                if hasattr(self.pipeline, 'query_engine'):
                    query_result = await self.pipeline.query_engine.query(self.last_transcription)
                    response = query_result.get("response", "")
                    
                    logger.info(f"Repeating response for: {self.last_transcription}")
                    
                    # Convert to speech
                    if response and hasattr(self.pipeline, 'tts_integration'):
                        try:
                            # Set agent speaking state before sending audio
                            self.set_agent_speaking(True)
                            self.speech_cancellation_event.clear()
                            
                            speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                            
                            # Convert to mulaw for Twilio
                            mulaw_audio = self._pcm_to_mulaw(speech_audio)
                            
                            # Send back to Twilio
                            logger.info(f"Sending repeated audio response ({len(mulaw_audio)} bytes)")
                            await self._send_audio(mulaw_audio, ws)
                            
                        except Exception as e:
                            logger.error(f"Error repeating response: {e}")
                        finally:
                            # Reset agent speaking state
                            self.set_agent_speaking(False)
        except Exception as e:
            logger.error(f"Error processing repeat request: {e}")
    
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
        
        # Add stronger filter for non-speech annotations - complete filter on noise-only transcriptions
        if re.search(r'^\(.*?\)$', text) or text.startswith('*') or text.endswith('*'):
            logger.info(f"Detected non-speech annotation, ignoring: '{text}'")
            return ""  # Return empty for parenthesized content
        
        # Remove noise keywords
        noise_keywords = ["click", "keyboard", "tongue", "scoff", "static", "clacking"]
        for keyword in noise_keywords:
            if keyword in text.lower():
                logger.info(f"Detected noise keyword '{keyword}', ignoring: '{text}'")
                return ""  # Return empty for noise content
        
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
        Check if a transcription is valid and worth processing with enhanced criteria.
        
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
        
        # Add noise keyword filtering
        noise_keywords = ["click", "keyboard", "tongue", "scoff", "static", "*"]
        if any(keyword in text.lower() for keyword in noise_keywords):
            logger.info(f"Transcription contains noise keywords: {text}")
            return False
        
        # Be more lenient with question marks and punctuation in confidence calculation
        confidence_estimate = 1.0
        if ("[" in text or "(" in text or "<" in text) and not "?" in text:
            confidence_estimate = 0.7  # Only reduce for annotation markers, not question marks
            logger.info(f"Reduced confidence due to uncertainty markers: {text}")
        
        if confidence_estimate < 0.65:
            logger.info(f"Transcription confidence too low: {confidence_estimate}")
            return False
        
        # Check word count - increased minimum requirement
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
        
        # Reset startup time
        self.startup_time = time.time()
        
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
        self.set_agent_speaking(False)  # Reset agent speaking state
        self.speech_cancellation_event.clear()
        self.silence_start_time = None
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.conversation_active = True
        self.stop_event.clear()
        self.startup_time = time.time()  # Reset startup time
        self.chunks_since_last_process = 0
        
        # Reset barge-in state
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        self.silence_frames = 0
        self.barge_in_detected = False
        
        # Send a welcome message with slight delay to give the client time to connect fully
        await asyncio.sleep(0.5)  # Small delay for connection setup
        await self.send_text_response("I'm listening. How can I help you today?", ws)
        
        # Initialize STT for streaming if available
        if hasattr(self.stt_integration, 'start_streaming'):
            try:
                # Check if it's an async method
                if asyncio.iscoroutinefunction(self.stt_integration.start_streaming):
                    await self.stt_integration.start_streaming()
                else:
                    # Handle if it's a synchronous method
                    self.stt_integration.start_streaming()
                logger.info("Started STT streaming session")
            except Exception as e:
                logger.error(f"Error starting STT streaming session: {e}")
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """
        Handle media event with audio data using Google Cloud STT.
        
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
            
            # Increment chunks counter
            self.chunks_since_last_process += 1
            
            # Check for startup delay period
            time_since_startup = time.time() - self.startup_time
            if time_since_startup < self.startup_delay_sec:
                logger.debug(f"In startup delay period ({time_since_startup:.1f}s < {self.startup_delay_sec:.1f}s)")
                return
            
            # Check for barge-in if agent is speaking
            if self.is_speaking:
                barge_in_detected = await self._check_for_barge_in(audio_data)
                if barge_in_detected:
                    # Log and signal barge-in
                    logger.info("Barge-in detected, interrupting agent speech")
                    self.speech_cancellation_event.set()
                    
                    # Add some delay to make interruption sound natural
                    await asyncio.sleep(0.3)  # 300ms debounce
                    
                    # Send silence to cut off current speech
                    silence_data = bytes([0x7f] * 320)  # 40ms of silence
                    await self._send_audio(silence_data, ws, is_interrupt=True)
                    
                    # Process new input immediately
                    async with self.processing_lock:
                        if not self.is_processing:
                            self.is_processing = True
                            try:
                                logger.info(f"Processing audio buffer after barge-in")
                                await self._process_audio(ws)
                            finally:
                                self.is_processing = False
                                self.chunks_since_last_process = 0
            
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
            if (len(self.input_buffer) >= AUDIO_BUFFER_SIZE and 
                not self.is_processing and 
                self.chunks_since_last_process >= 5):  # Only process every 5 chunks
                
                async with self.processing_lock:
                    if not self.is_processing:  # Double-check within lock
                        self.is_processing = True
                        try:
                            logger.info(f"Processing audio buffer of size: {len(self.input_buffer)} bytes")
                            await self._process_audio(ws)
                            self.chunks_since_last_process = 0  # Reset counter
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
        self.speech_cancellation_event.set()  # Cancel any ongoing speech
        
        # Cancel keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Close STT streaming session if needed
        if hasattr(self.stt_integration, 'stop_streaming'):
            try:
                await self.stt_integration.stop_streaming()
                logger.info("Stopped STT streaming session")
            except Exception as e:
                logger.error(f"Error stopping STT streaming session: {e}")
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """
        Handle mark event for audio playback tracking.
        
        Args:
            data: Mark event data
        """
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    async def _process_audio(self, ws) -> None:
        """
        Process accumulated audio data through Google Cloud STT and pipeline.
        
        Args:
            ws: WebSocket connection
        """
        try:
            # Convert Twilio mulaw buffer to PCM bytes
            mulaw_bytes = bytes(self.input_buffer)
            
            # Process through STT integration
            result = await self.stt_integration.transcribe_audio_data(mulaw_bytes)
            
            # Get transcription
            transcription = result.get("transcription", "")
            original_transcription = result.get("original_transcription", "")
            
            # Log transcriptions
            logger.info(f"RAW TRANSCRIPTION: '{original_transcription}'")
            logger.info(f"CLEANED TRANSCRIPTION: '{transcription}'")
            
            # Check for noise indicators or empty transcription
            if not transcription:
                logger.info("Empty transcription, skipping")
                # Clear part of buffer and continue
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
            
            # Check if this is a simple command before applying minimum word filter
            is_command, command_type = self._is_simple_command(transcription)
            if is_command:
                logger.info(f"Detected command: '{transcription}' -> {command_type}")
                
                # Clear input buffer since we recognized a command
                self.input_buffer.clear()
                
                # Handle the command
                command_handled = self._handle_command(command_type, ws)
                if command_handled:
                    return
            
            # Check minimum word count for regular queries (not commands)
            if not is_command and len(transcription.split()) < self.min_words_for_valid_query:
                logger.info(f"Transcription too short, skipping: '{transcription}'")
                # Clear part of buffer and continue
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
            
            # Process if it's a valid transcription or recognized command
            if transcription:
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
                        
                        # Make responses more concise for telephony
                        if len(response.split()) > 100:  # Long response
                            paragraphs = response.split("\n\n")
                            if len(paragraphs) > 2:
                                # Keep first two paragraphs and summarize
                                response = "\n\n".join(paragraphs[:2])
                                response += "\n\nI have more details if you'd like me to continue."
                        
                        logger.info(f"Generated response: {response}")
                        
                        # Convert to speech
                        if response and hasattr(self.pipeline, 'tts_integration'):
                            try:
                                # Set agent speaking state before sending audio
                                self.set_agent_speaking(True)
                                self.speech_cancellation_event.clear()
                                
                                speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                
                                # Convert to mulaw for Twilio
                                mulaw_audio = self._pcm_to_mulaw(speech_audio)
                                
                                # Send back to Twilio
                                logger.info(f"Sending audio response ({len(mulaw_audio)} bytes)")
                                await self._send_audio(mulaw_audio, ws)
                                
                            except Exception as tts_error:
                                logger.error(f"TTS Error: {tts_error}. Sending fallback response.", exc_info=True)
                                
                                # Try to send a text-based fallback response
                                fallback_message = "I'm having trouble generating speech at the moment. Let me try again in a moment."
                                
                                try:
                                    # Use a simple SSML template
                                    simple_audio = await self.pipeline.tts_integration.text_to_speech(fallback_message)
                                    fallback_mulaw = self._pcm_to_mulaw(simple_audio)
                                    await self._send_audio(fallback_mulaw, ws)
                                except Exception as fallback_error:
                                    logger.error(f"Failed to generate fallback response: {fallback_error}")
                            
                            finally:
                                # Reset agent speaking state
                                self.set_agent_speaking(False)
                            
                            # Update state
                            self.last_transcription = transcription
                            self.last_response_time = time.time()
                except Exception as e:
                    logger.error(f"Error processing through knowledge base: {e}", exc_info=True)
            else:
                # If no valid transcription, reduce buffer size but keep some for context
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                logger.debug(f"No valid transcription, reduced buffer to {len(self.input_buffer)} bytes")
            
            # Restart STT streaming session for next input
            if hasattr(self.stt_integration, 'start_streaming'):
                await self.stt_integration.start_streaming()
                logger.info("Reset STT streaming session for next input")
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            # If error, clear part of buffer and continue
            half_size = len(self.input_buffer) // 2
            self.input_buffer = self.input_buffer[half_size:]
    
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
            
            # Split audio into smaller chunks to avoid timeouts
            chunk_size = 320  # 20ms of 8kHz μ-law mono audio
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            
            logger.debug(f"Splitting {len(audio_data)} bytes into {len(chunks)} chunks")
            
            # Set agent speaking flag if this isn't an interrupt signal
            if not is_interrupt and not self.is_speaking:
                self.set_agent_speaking(True)
            
            for i, chunk in enumerate(chunks):
                # Check if we should cancel speech (barge-in detected)
                if self.speech_cancellation_event.is_set():
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
            if not is_interrupt and self.is_speaking:
                self.set_agent_speaking(False)
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=True)
            if "Connection closed" in str(e):
                self.connected = False
                self.connection_active.clear()
    
    def _update_ambient_noise_level(self, audio_data: bytes) -> None:
        """
        Update ambient noise level using adaptive statistics.
        
        Args:
            audio_data: Audio data as bytes
        """
        try:
            # Convert mu-law to PCM for energy calculation
            pcm_data = self._mulaw_to_pcm(audio_data)
            
            # Calculate energy for noise level
            import audioop
            rms = audioop.rms(pcm_data, 2)  # 2 bytes per sample for 16-bit PCM
            
            # Normalize to 0.0-1.0 range
            energy = rms / 32768.0
            
            # If audio is silence (very low energy), use it to update noise floor
            if energy < 0.02:  # Very quiet audio
                self.noise_samples.append(energy)
                # Keep only recent samples
                if len(self.noise_samples) > self.max_samples:
                    self.noise_samples.pop(0)
                
                # Update ambient noise level (with safety floor)
                if self.noise_samples:
                    # Use 90th percentile to avoid outliers
                    import numpy as np
                    new_noise_level = max(
                        self.min_noise_floor,  # Minimum threshold
                        np.percentile(self.noise_samples, 90) * 2.0  # Set threshold just above noise
                    )
                    
                    # Use exponential moving average to avoid abrupt changes
                    alpha = 0.3  # Weight for new value (0.3 = 30% new, 70% old)
                    self.ambient_noise_level = (alpha * new_noise_level + 
                                              (1 - alpha) * self.ambient_noise_level)
                    
                    # Update derived thresholds
                    self.low_threshold = self.ambient_noise_level * 2.0
                    self.high_threshold = self.ambient_noise_level * 4.5
        except Exception as e:
            logger.error(f"Error updating ambient noise level: {e}")
    
    def _calculate_speech_band_energy(self, audio_data: bytes) -> Dict[str, float]:
        """
        Calculate energy in specific frequency bands relevant to speech.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Dictionary with energy in different frequency bands
        """
        try:
            # Convert mu-law to PCM for spectral analysis
            pcm_data = self._mulaw_to_pcm(audio_data)
            
            # Convert to numpy array
            import numpy as np
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate FFT
            fft_data = np.abs(np.fft.rfft(audio_array))
            freqs = np.fft.rfftfreq(len(audio_array), 1/16000)  # 16kHz sample rate after conversion
            
            # Define frequency bands - focus on telephony speech range
            low_band_idx = (freqs < 300)
            speech_band_idx = (freqs >= 300) & (freqs <= 3400)  # Primary speech frequencies
            high_band_idx = (freqs > 3400)
            
            # Calculate energy in each band
            low_energy = np.sum(fft_data[low_band_idx]**2)  # Using squared magnitude for energy
            speech_energy = np.sum(fft_data[speech_band_idx]**2)
            high_energy = np.sum(fft_data[high_band_idx]**2)
            
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
    
    def _calculate_audio_energy(self, audio_data: bytes) -> float:
        """
        Calculate audio energy for basic speech detection.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Audio energy level
        """
        try:
            import audioop
            
            # Calculate RMS energy
            rms = audioop.rms(audio_data, 1)  # 1 byte per sample for mulaw
            
            # Normalize to 0.0-1.0 range
            normalized_rms = rms / 32768.0
            
            return normalized_rms
            
        except Exception as e:
            logger.error(f"Error calculating audio energy: {e}")
            return 0.0
    
    async def _check_for_barge_in(self, audio_data: bytes) -> bool:
        """
        Enhanced barge-in detection with state machine.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            True if barge-in is detected
        """
        # Don't check for barge-in if we're not speaking
        if not self.is_speaking:
            return False
            
        try:
            # Update noise floor for adaptive thresholds
            self._update_ambient_noise_level(audio_data)
            
            # Calculate energy of the audio
            energy = self._calculate_audio_energy(audio_data)
            
            # Get speech band energy for better detection
            band_energies = self._calculate_speech_band_energy(audio_data)
            speech_ratio = band_energies.get("speech_band", 0.0)
            
            # Calculate the average speech band ratio from history
            avg_speech_ratio = 0.4  # Default
            if self.speech_band_energies:
                import numpy as np
                avg_speech_ratio = np.mean(self.speech_band_energies)
            
            import time
            current_time = time.time() * 1000  # Convert to ms
            
            # Check cooldown period - don't detect barge-in too early after we start speaking
            time_since_agent_started = current_time - (self.agent_speaking_start_time * 1000)
            if time_since_agent_started < self.barge_in_cooldown_ms:  # E.g., 1.5 second cooldown
                return False
            
            # Check if enough time has passed since last barge-in
            time_since_last_barge_in = current_time - (self.last_barge_in_time * 1000)
            if time_since_last_barge_in < 800:  # 800ms minimum between barge-ins
                return False
                
            # State machine for more robust barge-in detection
            if self.speech_state == SpeechState.SILENCE:
                # Check if potential speech detected - more strict criteria for barge-in
                if energy > self.low_threshold and speech_ratio > 0.6:
                    self.speech_state = SpeechState.POTENTIAL_SPEECH
                    self.potential_speech_frames = 1
                    logger.debug(f"Potential barge-in speech detected: energy={energy:.4f}, speech_ratio={speech_ratio:.2f}")
                    return False  # Not confirmed yet
                return False  # Still silence
                
            elif self.speech_state == SpeechState.POTENTIAL_SPEECH:
                # Check if still potential speech
                if energy > self.low_threshold and speech_ratio > 0.6:
                    self.potential_speech_frames += 1
                    # Check if we have enough frames for barge-in - require fewer frames
                    if self.potential_speech_frames >= 3:  # Need 3 consecutive frames
                        # Check against higher threshold for confirmation
                        if energy > self.high_threshold and speech_ratio > 0.65:
                            self.speech_state = SpeechState.CONFIRMED_SPEECH
                            self.confirmed_speech_frames = 1
                            
                            # Set barge-in detected state
                            self.barge_in_detected = True
                            self.last_barge_in_time = time.time()
                            
                            logger.info(f"Barge-in confirmed: energy={energy:.4f}, speech_ratio={speech_ratio:.2f}")
                            return True
                    return False  # Not confirmed yet
                else:
                    # Go back to silence
                    self.speech_state = SpeechState.SILENCE
                    self.potential_speech_frames = 0
                    return False
                    
            elif self.speech_state == SpeechState.CONFIRMED_SPEECH:
                # Already confirmed speech, maintain barge-in signal
                self.confirmed_speech_frames += 1
                return True
                
            # Default check in case we're in an unexpected state
            # More aggressive barge-in detection during agent speech
            # Use lower threshold (80% of normal) when agent is speaking
            barge_in_threshold_adjusted = self.barge_in_threshold * 0.8
            
            # Look for sudden energy spike which is common in interruptions
            is_energy_spike = energy > barge_in_threshold_adjusted
            # Look for good speech ratio which indicates speech rather than noise
            is_good_speech_quality = speech_ratio > 0.55
            
            if is_energy_spike and is_good_speech_quality:
                # Update barge-in state
                self.barge_in_detected = True
                self.last_barge_in_time = time.time()
                
                logger.info(f"Barge-in detected (direct): energy={energy:.4f}, speech_ratio={speech_ratio:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in barge-in detection: {e}")
            return False
    
    def _pcm_to_mulaw(self, pcm_data: bytes) -> bytes:
        """
        Convert PCM audio to mulaw for Twilio.
        
        Args:
            pcm_data: PCM audio data
            
        Returns:
            Mulaw audio data
        """
        try:
            import audioop
            
            # Check for extremely small data
            if len(pcm_data) < 320:  # Less than 20ms at 8kHz
                # Pad tiny chunks to avoid warnings
                padding_size = 320 - len(pcm_data)
                pcm_data = pcm_data + (b'\x00' * padding_size)
            
            # Ensure we have an even number of bytes for 16-bit samples
            if len(pcm_data) % 2 != 0:
                pcm_data = pcm_data + b'\x00'
            
            # Resample to 8kHz for Twilio
            pcm_data_8k, _ = audioop.ratecv(
                pcm_data, 2, 1, 
                16000,  # Assuming 16kHz input
                8000,   # Twilio expects 8kHz
                None
            )
            
            # Convert to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_data_8k, 2)
            
            # Ensure minimum size for mulaw data
            if len(mulaw_data) < 160:  # Minimum 20ms chunk
                padding_needed = 160 - len(mulaw_data)
                mulaw_data = mulaw_data + bytes([0x7f] * padding_needed)  # Use 0x7f (silence) for padding
            
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error converting PCM to mulaw: {e}", exc_info=True)
            # Return silence rather than empty data
            return bytes([0x7f] * 160)  # Return 20ms of silence
    
    def _mulaw_to_pcm(self, mulaw_data: bytes) -> bytes:
        """
        Convert Twilio's mulaw audio to PCM.
        
        Args:
            mulaw_data: Audio data in mulaw format
            
        Returns:
            Audio data as PCM bytes
        """
        try:
            import audioop
            
            # Convert mulaw to 16-bit PCM
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Resample from 8kHz to 16kHz for Google Cloud STT
            pcm_data_16k, _ = audioop.ratecv(
                pcm_data, 2, 1, 
                8000,  # Twilio uses 8kHz 
                16000, # Google Cloud STT expects 16kHz
                None
            )
            
            return pcm_data_16k
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            return b''
    
    def set_agent_speaking(self, is_speaking: bool) -> None:
        """
        Set the agent speaking state for barge-in detection.
        
        Args:
            is_speaking: Whether the agent is speaking
        """
        # Update speaking state
        self.is_speaking = is_speaking
        self.agent_speaking = is_speaking
        
        # Track speech start time for cooldown
        if is_speaking:
            import time
            self.agent_speaking_start_time = time.time()
            self.speech_cancellation_event.clear()
            
            # Reset speech state for new detection
            self.speech_state = SpeechState.SILENCE
            self.potential_speech_frames = 0
            self.confirmed_speech_frames = 0
        
        # Try to update STT integration's speech detector if available
        if hasattr(self.stt_integration, 'speech_detector') and self.stt_integration.speech_detector is not None:
            try:
                self.stt_integration.speech_detector.set_agent_speaking(is_speaking)
            except Exception as e:
                logger.warning(f"Unable to update STT integration speech detector: {e}")
    
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
                self.set_agent_speaking(True)
                self.speech_cancellation_event.clear()
                
                speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                
                # Convert to mulaw
                mulaw_audio = self._pcm_to_mulaw(speech_audio)
                
                # Send audio
                await self._send_audio(mulaw_audio, ws)
                logger.info(f"Sent text response: '{text}'")
                
                # Reset agent speaking state
                self.set_agent_speaking(False)
                
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