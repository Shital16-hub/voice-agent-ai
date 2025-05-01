"""
Enhanced WebSocket handler for Twilio media streams with improved speech/noise
discrimination and barge-in detection using the new AudioPreprocessor.
This version has optimized Alexa-like conversation flow and improved barge-in functionality.
"""
import json
import base64
import asyncio
import logging
import time
import numpy as np
import re
import audioop
from typing import Dict, Any, Callable, Awaitable, Optional, List, Union, Tuple
from collections import deque
from enum import Enum

from telephony.audio_preprocessor import AudioPreprocessor, SpeechState
from telephony.config import (
    CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, 
    MAX_BUFFER_SIZE, ENABLE_BARGE_IN, BARGE_IN_THRESHOLD, BARGE_IN_DETECTION_WINDOW
)

# Import Google Cloud STT integration instead of Deepgram
from speech_to_text.google_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

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
]

class ConversationState(Enum):
    IDLE = "idle"  # Not actively engaged
    LISTENING = "listening"  # Listening for user input
    PROCESSING = "processing"  # Processing user input
    RESPONDING = "responding"  # Agent is speaking
    PAUSED = "paused"  # Conversation paused

class WebSocketHandler:
    """
    Enhanced WebSocket handler for Twilio media streams with improved barge-in detection
    and Alexa-like conversation flow.
    """
    
    # Enhanced command lists
    STOP_COMMANDS = [
        "stop", "stop speaking", "stop talking", "be quiet", "shut up", "quiet", 
        "enough", "halt", "pause", "quit", "exit", "bye", "end", "silence"
    ]

    CONTINUE_COMMANDS = [
        "continue", "go on", "proceed", "resume", "keep going", "carry on", "go ahead"
    ]

    REPEAT_COMMANDS = [
        "repeat", "say again", "repeat that", "what did you say", "say that again", 
        "can you repeat", "again", "what was that"
    ]

    HELP_COMMANDS = [
        "help", "help me", "what can you do", "how does this work", "instructions", 
        "commands", "options"
    ]
    
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
        self.audio_processor = pipeline.speech_recognizer.audio_processor if hasattr(pipeline.speech_recognizer, 'audio_processor') else None
        
        # Initialize audio processor if not already present
        if not self.audio_processor:
            from telephony.audio_processor import AudioProcessor
            self.audio_processor = AudioProcessor()
        
        # Audio buffers - reduced size for lower latency
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
        
        # Add variables for echo detection
        self.agent_last_response = ""  # Store the agent's last response text
        self.echo_detection_window = 3.0  # Seconds to ignore similar utterances after agent speaks (reduced)
        self.similarity_threshold = 0.5  # Threshold for considering text similar (reduced)
        self.last_agent_speech_time = 0.0  # When the agent last finished speaking
        self.is_processing_self_echo = False  # Flag to track potential self-echoes
        
        # Conversation state management
        self.conversation_state = ConversationState.IDLE
        self.last_state_change_time = time.time()
        self.max_state_duration = {
            ConversationState.IDLE: 30.0,  # 30 seconds max in idle (reduced)
            ConversationState.LISTENING: 5.0,  # 5 seconds max listening (reduced)
            ConversationState.PROCESSING: 3.0,  # 3 seconds max processing (reduced)
            ConversationState.RESPONDING: 15.0,  # 15 seconds max responding (reduced)
            ConversationState.PAUSED: 15.0  # 15 seconds max paused (reduced)
        }
        
        # Track state timeout tasks
        self.state_timeout_task = None
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Conversation flow management with increased pause
        self.pause_after_response = 0.8  # Reduced from 1.5 seconds for more responsive flow
        self.min_words_for_valid_query = 1  # Reduced to 1 word minimum for higher responsiveness
        
        # Create an event to signal when we should stop processing
        self.stop_event = asyncio.Event()
        
        # Barge-in handling
        self.speech_cancellation_event = asyncio.Event()
        self.speech_cancellation_event.clear()
        
        # New: Add startup delay before processing audio (prevents false triggers at start)
        self.startup_time = time.time()
        self.startup_delay_sec = 0.5  # Reduced to 0.5 second for faster initial response
        
        # New: Track consecutive silent frames for better silence detection
        self.consecutive_silent_frames = 0
        self.min_silent_frames_for_silence = 3  # Reduced for faster response
        
        # New: Add a flag to prevent multiple barge-in detections in quick succession
        self.last_barge_in_time = 0
        self.barge_in_cooldown_sec = 0.3  # Reduced to 0.3 seconds for faster interruption
        
        # Track chunks processed since last processing 
        self.chunks_since_last_process = 0
        
        # Reduced buffer size for lower latency
        self.buffer_size_minimum = AUDIO_BUFFER_SIZE // 4  # Quarter the original size

        # Conversation manager reference
        if hasattr(self.pipeline, 'conversation_manager'):
            self.conversation_manager = self.pipeline.conversation_manager
        else:
            self.conversation_manager = None
        
        # Add flag for "Alexa-like" wait indicator
        self.has_played_listening_indicator = False
        
        # Add flag to track first interaction to avoid command detection
        self.first_interaction = True
        
        # Add confidence threshold for command detection
        self.min_command_confidence = 0.8
        
        logger.info(f"Enhanced WebSocketHandler initialized for call {call_sid} with improved barge-in")
    
    def _set_conversation_state(self, new_state: ConversationState) -> None:
        """
        Set conversation state with enhanced conversational flow.
        
        Args:
            new_state: New conversation state
        """
        old_state = self.conversation_state
        if old_state == new_state:
            return  # No change
        
        # Update state
        self.conversation_state = new_state
        self.last_state_change_time = time.time()
        
        # Log state change
        logger.info(f"Conversation state change: {old_state.value} -> {new_state.value}")
        
        # Cancel existing timeout task
        if self.state_timeout_task:
            self.state_timeout_task.cancel()
        
        # Create new timeout task
        max_duration = self.max_state_duration.get(new_state, 30.0)
        self.state_timeout_task = asyncio.create_task(self._handle_state_timeout(new_state, max_duration))
        
        # Perform state-specific actions
        if new_state == ConversationState.LISTENING:
            # Reset listening indicator flag
            self.has_played_listening_indicator = False
            
            # Ensure STT is ready
            if hasattr(self.pipeline, 'speech_recognizer'):
                if hasattr(self.pipeline.speech_recognizer, 'start_streaming'):
                    if not getattr(self.pipeline.speech_recognizer, 'is_streaming', False):
                        asyncio.create_task(self.pipeline.speech_recognizer.start_streaming())
            
            # Add a subtle audio indicator for listening state
            if old_state == ConversationState.RESPONDING:
                # Play a very short "ready" tone if we just finished speaking
                asyncio.create_task(self._play_listening_indicator())
        
        elif new_state == ConversationState.RESPONDING:
            # Set agent speaking flag
            if hasattr(self.audio_processor, 'set_agent_speaking'):
                self.audio_processor.set_agent_speaking(True)
                
        elif new_state == ConversationState.IDLE or new_state == ConversationState.LISTENING:
            # Clear agent speaking flag if transitioning from RESPONDING
            if old_state == ConversationState.RESPONDING and hasattr(self.audio_processor, 'set_agent_speaking'):
                self.audio_processor.set_agent_speaking(False)

    async def _handle_state_timeout(self, expected_state: ConversationState, timeout: float) -> None:
        """
        Handle timeout for a conversation state.
        
        Args:
            expected_state: The state that should timeout
            timeout: Timeout duration in seconds
        """
        try:
            await asyncio.sleep(timeout)
            
            # Only act if we're still in the same state
            if self.conversation_state == expected_state:
                logger.warning(f"Conversation state {expected_state.value} timed out after {timeout}s")
                
                # Handle timeout based on state
                if expected_state == ConversationState.RESPONDING:
                    # Force stop speaking
                    self.speech_cancellation_event.set()
                    self._set_conversation_state(ConversationState.LISTENING)
                    
                elif expected_state == ConversationState.PROCESSING:
                    # Reset processing state
                    self.is_processing = False
                    self._set_conversation_state(ConversationState.LISTENING)
                    
                elif expected_state == ConversationState.LISTENING:
                    # If listening too long without input, go to idle
                    self._set_conversation_state(ConversationState.IDLE)
                    
                elif expected_state == ConversationState.PAUSED:
                    # Resume from pause after timeout
                    self._set_conversation_state(ConversationState.LISTENING)
                    
        except asyncio.CancelledError:
            # Task was cancelled, this is expected when state changes
            pass
        except Exception as e:
            logger.error(f"Error in state timeout handler: {e}")

    def _detect_barge_in(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced barge-in detection that works directly with audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if barge-in is detected
        """
        if not self.audio_processor.preprocessor.agent_speaking:
            return False  # Only check for barge-in when agent is speaking
        
        # Calculate energy of the audio
        energy = np.mean(np.abs(audio_data))
        
        # Use a more sensitive threshold for barge-in detection
        barge_in_threshold = 0.02  # Decreased from 0.03 for better sensitivity
        
        # Check for current time since agent started speaking
        current_time = time.time()
        time_since_agent_started = (current_time - self.audio_processor.preprocessor.agent_speaking_start_time) * 1000  # ms
        
        # Reduced delay before allowing barge-in
        if time_since_agent_started < 500:  # Reduced from 800ms to 500ms
            return False
        
        # Check time since last barge-in to avoid multiple triggers
        time_since_last_barge_in = (current_time - self.last_barge_in_time) * 1000  # ms
        if time_since_last_barge_in < 300:  # Reduced from 500ms to 300ms
            return False
        
        # Check energy threshold
        if energy > barge_in_threshold:
            # Update last barge-in time
            self.last_barge_in_time = current_time
            logger.info(f"Barge-in detected! Energy: {energy:.4f}, Threshold: {barge_in_threshold:.4f}")
            return True
        
        return False
    
    def _is_similar_to_last_response(self, transcription: str) -> bool:
        """
        Check if a transcription is similar to the agent's last response (indicating echo).
        
        Args:
            transcription: Transcription text to check
            
        Returns:
            True if transcription is likely an echo of agent's speech
        """
        if not self.agent_last_response or not transcription:
            return False
        
        # Simple word-overlap similarity check
        last_response_words = set(self.agent_last_response.lower().split())
        transcription_words = set(transcription.lower().split())
        
        # If too few words to check meaningfully, be conservative
        if len(transcription_words) < 2:
            return False
        
        # Calculate overlap
        intersection = last_response_words.intersection(transcription_words)
        
        if len(transcription_words) == 0:
            return False
            
        # Calculate Jaccard similarity
        similarity = len(intersection) / len(transcription_words)
        
        # Time-based factor - more strict just after agent spoke, more lenient as time passes
        time_since_agent_speech = time.time() - self.last_agent_speech_time
        time_factor = max(0.5, min(1.0, time_since_agent_speech / self.echo_detection_window))
        
        # Adjust threshold based on time
        effective_threshold = self.similarity_threshold * time_factor
        
        # Log for debugging
        if similarity > 0.3:  # Only log potentially problematic cases
            logger.debug(f"Echo check: Similarity {similarity:.2f}, Time factor {time_factor:.2f}, " 
                        f"Effective threshold {effective_threshold:.2f}")
        
        # Short transcriptions with high word match are likely echoes
        if similarity > effective_threshold:
            logger.info(f"Detected potential echo: '{transcription}' (similarity: {similarity:.2f})")
            return True
        
        return False
    
    def _is_simple_command(self, text: str) -> Tuple[bool, str]:
        """
        Improved method to check if text is a command with better matching for stop commands.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_command, command_type)
        """
        if not text:
            return False, ""
        
        # Clean and lowercase the text
        clean_text = text.lower().strip().rstrip('.!?')
        
        # Special check for stop command with higher priority
        stop_commands = ["stop", "stop speaking", "stop talking", "be quiet", "shut up", "quiet", 
                        "enough", "halt", "pause", "quit", "exit", "bye", "end", "silence"]
        
        # Check if any word in the text matches stop commands exactly
        words = clean_text.split()
        for word in words:
            if word in ["stop", "quiet", "silence", "halt", "pause"]:
                logger.info(f"Detected stop command: '{clean_text}' -> stop")
                return True, "stop"
        
        # Define all command categories
        command_categories = {
            "stop": self.STOP_COMMANDS,
            "continue": self.CONTINUE_COMMANDS,
            "repeat": self.REPEAT_COMMANDS,
            "help": self.HELP_COMMANDS
        }
        
        # Check each category
        for command_type, commands in command_categories.items():
            # Direct exact matches
            if clean_text in commands:
                logger.info(f"Detected exact command: '{clean_text}' -> {command_type}")
                return True, command_type
            
            # Check if any command is a substring of the text
            for cmd in commands:
                if f" {cmd} " in f" {clean_text} " or clean_text.startswith(cmd + " ") or clean_text.endswith(" " + cmd):
                    logger.info(f"Detected command in phrase: '{clean_text}' -> {command_type}")
                    return True, command_type
            
            # Check for partial word matches with higher threshold for shorter commands
            for cmd in commands:
                cmd_tokens = cmd.split()
                
                # Skip short commands like "stop" for partial matching to avoid false positives
                if len(cmd) < 5 and len(cmd_tokens) == 1:
                    continue
                    
                # For multi-word commands, check if most words match
                if len(cmd_tokens) > 1 and len(words) >= len(cmd_tokens):
                    # FIXED: Use exact word matching instead of substring matching
                    matches = sum(1 for cmd_token in cmd_tokens if cmd_token in words)
                    if matches >= len(cmd_tokens) * 0.7:  # 70% of words match
                        logger.info(f"Detected partial command match: '{clean_text}' -> {command_type}")
                        return True, command_type
        
        return False, ""
    
    def _handle_command(self, command_type: str, ws) -> bool:
        """
        Improved command handling with better user feedback.
        
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
            
            # Set to paused state
            self._set_conversation_state(ConversationState.PAUSED)
            
            # Send confirmation to user - shorter acknowledgment
            asyncio.create_task(self.send_text_response("Okay, stopping.", ws))
            
            # Reset buffer
            self.input_buffer.clear()
            return True
            
        elif command_type == "continue":
            # Acknowledge continue command and resume listening
            logger.info("Continue command detected")
            self._set_conversation_state(ConversationState.LISTENING)
            asyncio.create_task(self.send_text_response("I'll continue.", ws))
            return True
            
        elif command_type == "help":
            # Provide help - more concise and informative
            logger.info("Help command detected")
            help_text = ("You can ask me questions about VoiceAI and its features. "
                        "You can also say 'stop' to interrupt me, 'repeat' to hear something again, "
                        "or 'continue' to resume.")
            asyncio.create_task(self.send_text_response(help_text, ws))
            return True
            
        elif command_type == "repeat":
            # Repeat last response with clear acknowledgment
            logger.info("Repeat command detected")
            if hasattr(self, 'last_transcription') and hasattr(self, 'last_response_time'):
                if self.last_transcription:
                    # First acknowledge
                    asyncio.create_task(self.send_text_response("Sure, I'll repeat that.", ws))
                    # Add a small delay before repeating
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
                    logger.info(f"Repeating response for: {self.last_transcription}")
                    query_result = await self.pipeline.query_engine.query(self.last_transcription)
                    response = query_result.get("response", "")
                    
                    # Convert to speech
                    if response and hasattr(self.pipeline, 'tts_integration'):
                        try:
                            # Set agent speaking state before sending audio
                            self.audio_processor.set_agent_speaking(True)
                            self.speech_cancellation_event.clear()
                            
                            speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                            
                            # Convert to mulaw for Twilio
                            mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                            
                            # Send back to Twilio
                            logger.info(f"Sending repeated audio response ({len(mulaw_audio)} bytes)")
                            await self._send_audio(mulaw_audio, ws)
                            
                        except Exception as e:
                            logger.error(f"Error repeating response: {e}")
                        finally:
                            # Reset agent speaking state
                            self.audio_processor.set_agent_speaking(False)
        except Exception as e:
            logger.error(f"Error processing repeat request: {e}")
    
    def cleanup_transcription(self, text: str) -> str:
        """
        More lenient cleanup of transcription text.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
            
        # Keep simple cleanup but reduce filtering
        cleaned_text = text
        
        # Only remove obvious noise markers
        cleaned_text = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Log what was cleaned if substantial
        if text != cleaned_text:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text
    
    async def _play_listening_indicator(self):
        """Play a subtle audio indicator when the system is ready to listen."""
        if self.has_played_listening_indicator:
            return  # Don't play multiple times
            
        try:
            # Create a very short, subtle "ready" tone using mulaw encoding
            # Generate a short beep using simple synthesis
            ready_tone = bytes([0x7f, 0x8f, 0x9f, 0xaf, 0xbf, 0xcf, 0xdf, 0xef, 0xff,
                            0xef, 0xdf, 0xcf, 0xbf, 0xaf, 0x9f, 0x8f, 0x7f]) * 2
            
            # Pad with silence
            silence = bytes([0x7f] * 10)
            ready_signal = silence + ready_tone + silence
            
            # Send the tone if we have an active WebSocket connection
            if hasattr(self, 'connected') and self.connected and hasattr(self, 'stream_sid') and self.stream_sid:
                # Find the active WebSocket
                if hasattr(self, 'ws') and self.ws:
                    await self._send_audio(ready_signal, self.ws, is_indicator=True)
                    self.has_played_listening_indicator = True
                    logger.info("Played listening indicator sound")
        except Exception as e:
            logger.error(f"Error playing listening indicator: {e}")
    
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
            
            # Store the WebSocket connection for use in _play_listening_indicator
            self.ws = ws
            
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
        self.audio_processor.set_agent_speaking(False)  # Reset agent speaking state
        self.speech_cancellation_event.clear()
        self.silence_start_time = None
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.conversation_active = True
        self.stop_event.clear()
        self.startup_time = time.time()  # Reset startup time
        self.consecutive_silent_frames = 0
        self.last_barge_in_time = 0
        self.chunks_since_last_process = 0
        self.has_played_listening_indicator = False
        self.first_interaction = True  # Reset first interaction flag
        
        # Reset the audio processor
        self.audio_processor.reset()
        
        # Initialize conversation state
        self._set_conversation_state(ConversationState.IDLE)
        
        # Send a welcome message with slight delay to give the client time to connect fully
        await asyncio.sleep(0.5)  # Small delay for connection setup
        await self.send_text_response("I'm listening. How can I help you today?", ws)
        
        # Initialize STT for streaming if available
        if hasattr(self.pipeline, 'speech_recognizer'):
            try:
                # Check if the speech recognizer has start_streaming method
                if hasattr(self.pipeline.speech_recognizer, 'start_streaming'):
                    # Check if it's an async method
                    if asyncio.iscoroutinefunction(self.pipeline.speech_recognizer.start_streaming):
                        await self.pipeline.speech_recognizer.start_streaming()
                    else:
                        # Handle if it's a synchronous method
                        self.pipeline.speech_recognizer.start_streaming()
                    logger.info("Started STT streaming session")
            except Exception as e:
                logger.error(f"Error starting STT streaming session: {e}")
    
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
        
        # Only process media if in IDLE, LISTENING or RESPONDING states
        if self.conversation_state not in [ConversationState.IDLE, ConversationState.LISTENING, ConversationState.RESPONDING]:
            logger.debug(f"Not processing media in {self.conversation_state.value} state")
            return
        
        # If in IDLE, move to LISTENING when receiving media
        if self.conversation_state == ConversationState.IDLE:
            self._set_conversation_state(ConversationState.LISTENING)
            
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
            
            # Check for barge-in immediately if agent is speaking
            if self.audio_processor.preprocessor.agent_speaking:
                # Convert mulaw to PCM for energy detection
                try:
                    pcm_audio = self.audio_processor.mulaw_to_pcm(audio_data)
                    if isinstance(pcm_audio, bytes):
                        audio_array = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        audio_array = pcm_audio
                        
                    # Check for barge-in with the simplified detector
                    if self._detect_barge_in(audio_array):
                        # Signal barge-in
                        self.speech_cancellation_event.set()
                        
                        # Add some delay for natural interruption
                        await asyncio.sleep(0.1)
                        
                        # Send silence to cut off current speech
                        silence_data = bytes([0x7f] * 320)  # 40ms of silence
                        await self._send_audio(silence_data, ws, is_interrupt=True)
                        
                        # Process buffer immediately
                        await self._process_audio(ws)
                        return  # Skip further processing for this chunk
                except Exception as e:
                    logger.error(f"Error in barge-in detection: {e}")
            
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
            # Use reduced buffer size for lower latency
            if (len(self.input_buffer) >= self.buffer_size_minimum and 
                not self.is_processing and 
                (self.audio_processor.preprocessor.speech_state == SpeechState.CONFIRMED_SPEECH or 
                 self.chunks_since_last_process >= 3)):  # Only process every 3 chunks if no speech (reduced)
                
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
        if hasattr(self.pipeline, 'speech_recognizer'):
            try:
                if hasattr(self.pipeline.speech_recognizer, 'stop_streaming'):
                    await self.pipeline.speech_recognizer.stop_streaming()
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
        Simplified process for accumulated audio data through the pipeline with Google Cloud Speech-to-Text.
        Bypasses extensive preprocessing and avoids unnecessary STT session resets.
        
        Args:
            ws: WebSocket connection
        """
        # Set state to processing
        self._set_conversation_state(ConversationState.PROCESSING)
        
        try:
            # Basic conversion without extensive preprocessing
            mulaw_bytes = bytes(self.input_buffer)
            
            try:
                # Simple conversion to PCM
                pcm_data = audioop.ulaw2lin(mulaw_bytes, 2)
                pcm_audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Basic noise gate - very mild
                noise_threshold = 0.005  # Reduced threshold
                pcm_audio = np.where(np.abs(pcm_audio) < noise_threshold, 0, pcm_audio)
                
                # Simple normalization
                if np.max(np.abs(pcm_audio)) > 0:
                    pcm_audio = pcm_audio / np.max(np.abs(pcm_audio)) * 0.95
                    
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                # Clear part of buffer and try again next time
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                self._set_conversation_state(ConversationState.LISTENING)
                return
                    
            # Create a list to collect transcription results
            transcription_results = []
            
            # Define a callback to collect results
            async def transcription_callback(result):
                # Store all results - we'll filter later
                transcription_results.append(result)
                logger.debug(f"Received transcription result: {result.text if hasattr(result, 'text') else 'No text'}")
            
            # Process audio through STT
            try:
                # Ensure STT is properly initialized and in streaming mode
                if hasattr(self.pipeline, 'speech_recognizer'):
                    # Make sure streaming is started
                    if hasattr(self.pipeline.speech_recognizer, 'is_streaming'):
                        if not getattr(self.pipeline.speech_recognizer, 'is_streaming', False):
                            if hasattr(self.pipeline.speech_recognizer, 'start_streaming'):
                                await self.pipeline.speech_recognizer.start_streaming()
                                await asyncio.sleep(0.1)  # Small delay
                
                # Convert to bytes format for Google Cloud
                audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
                
                # Process through Google Cloud Speech-to-Text
                # Google Cloud expects 16-bit PCM audio
                await self.pipeline.speech_recognizer.process_audio_chunk(
                    audio_chunk=audio_bytes,
                    callback=transcription_callback
                )
                
                # Check for results in the queue
                while not self.pipeline.speech_recognizer.result_queue.empty():
                    try:
                        result = self.pipeline.speech_recognizer.result_queue.get_nowait()
                        if result and result not in transcription_results:
                            transcription_results.append(result)
                            self.pipeline.speech_recognizer.result_queue.task_done()
                    except Exception:
                        break
                
                # Find any final results
                final_results = [r for r in transcription_results if 
                                getattr(r, 'is_final', False) and 
                                hasattr(r, 'text') and r.text]
                
                # Get best result
                best_result = None
                if final_results:
                    # Use result with highest confidence if available
                    if hasattr(final_results[0], 'confidence'):
                        best_result = max(final_results, key=lambda r: getattr(r, 'confidence', 0))
                    else:
                        # Otherwise use longest text
                        best_result = max(final_results, key=lambda r: len(getattr(r, 'text', '')))
                
                # Get transcription from best result
                transcription = best_result.text if best_result and hasattr(best_result, 'text') else ""
                
                # Check for commands first - do this before echo detection
                if transcription:
                    logger.info(f"Transcription: '{transcription}'")
                    
                    # Skip command detection on first interaction
                    # FIXED: Skip command detection for first interaction to avoid false command recognition
                    is_command, command_type = (False, "") if self.first_interaction else self._is_simple_command(transcription)
                    if self.first_interaction:
                        self.first_interaction = False
                    
                    if is_command:
                        # Handle command and return early - don't process as regular query
                        logger.info(f"Handling command: {command_type}")
                        
                        # Clear the input buffer since we're handling a command
                        self.input_buffer.clear()
                        
                        # Handle the command
                        if self._handle_command(command_type, ws):
                            # Command was handled, update state and return early
                            self.last_transcription = transcription
                            self.last_response_time = time.time()
                            return
                
                    # Check for echo - if within time window and similar to last response
                    time_since_agent_speech = time.time() - self.last_agent_speech_time
                    if (time_since_agent_speech < self.echo_detection_window and 
                        self._is_similar_to_last_response(transcription)):
                        logger.info(f"Ignoring likely self-echo: '{transcription}'")
                        # Reduce buffer but don't clear
                        half_size = len(self.input_buffer) // 2
                        self.input_buffer = self.input_buffer[half_size:]
                        self._set_conversation_state(ConversationState.LISTENING)
                        return  # Skip further processing
                    
                    # Process if it's a valid transcription - more lenient (1 word minimum)
                    if len(transcription.split()) >= self.min_words_for_valid_query:
                        # Now clear the input buffer since we have a valid transcription
                        self.input_buffer.clear()
                        
                        # Don't process duplicate transcriptions
                        if transcription == self.last_transcription:
                            logger.info("Duplicate transcription, not processing again")
                            self._set_conversation_state(ConversationState.LISTENING)
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
                                        # Set state to responding before sending audio
                                        self._set_conversation_state(ConversationState.RESPONDING)
                                        
                                        # Clear the speech cancellation event
                                        self.speech_cancellation_event.clear()
                                        
                                        speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                        
                                        # Convert to mulaw for Twilio
                                        mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                                        
                                        # Send back to Twilio
                                        logger.info(f"Sending audio response ({len(mulaw_audio)} bytes)")
                                        await self._send_audio(mulaw_audio, ws)
                                        
                                    except Exception as tts_error:
                                        logger.error(f"TTS Error: {tts_error}", exc_info=True)
                                        # Try to send a text-based fallback response
                                        fallback_message = "I'm having trouble generating speech right now."
                                        
                                        try:
                                            # Use a simple message
                                            simple_audio = await self.pipeline.tts_integration.tts_client.synthesize(
                                                fallback_message, is_ssml=False
                                            )
                                            fallback_mulaw = self.audio_processor.pcm_to_mulaw(simple_audio)
                                            await self._send_audio(fallback_mulaw, ws)
                                        except Exception as fallback_error:
                                            logger.error(f"Failed to generate fallback: {fallback_error}")
                                    finally:
                                        # Set back to listening
                                        self._set_conversation_state(ConversationState.LISTENING)
                                    
                                    # Update state
                                    self.last_transcription = transcription
                                    self.last_response_time = time.time()
                        except Exception as e:
                            logger.error(f"Error processing through knowledge base: {e}", exc_info=True)
                            # On error, go back to listening
                            self._set_conversation_state(ConversationState.LISTENING)
                    else:
                        # If no valid transcription, reduce buffer size but keep some for context
                        half_size = len(self.input_buffer) // 2
                        self.input_buffer = self.input_buffer[half_size:]
                        logger.debug(f"Transcription too short, reduced buffer to {len(self.input_buffer)} bytes")
                        self._set_conversation_state(ConversationState.LISTENING)
                else:
                    # If no transcription at all, just reduce buffer size
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    logger.debug(f"No transcription found, reduced buffer to {len(self.input_buffer)} bytes")
                    self._set_conversation_state(ConversationState.LISTENING)
                
                # Reset the STT session only if we got a valid transcription
                if transcription and len(transcription.split()) >= self.min_words_for_valid_query:
                    try:
                        if hasattr(self.pipeline.speech_recognizer, 'stop_streaming'):
                            await self.pipeline.speech_recognizer.stop_streaming()
                        
                        await asyncio.sleep(0.1)  # Add a smaller delay between stop and start
                        
                        if hasattr(self.pipeline.speech_recognizer, 'start_streaming'):
                            await self.pipeline.speech_recognizer.start_streaming()
                            await asyncio.sleep(0.1)  # Add a small delay
                        
                        logger.info("Reset STT after successful transcription")
                    except Exception as reset_error:
                        logger.error(f"Error resetting STT: {reset_error}")
            
            except Exception as e:
                logger.error(f"Error during STT processing: {e}", exc_info=True)
                # If error, clear part of buffer and continue
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                self._set_conversation_state(ConversationState.LISTENING)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            # On error, go back to listening
            self._set_conversation_state(ConversationState.LISTENING)
    
    async def _send_audio(self, audio_data: bytes, ws, is_interrupt: bool = False, is_indicator: bool = False) -> None:
        """
        Send audio data to Twilio with immediate barge-in interruption support.
        
        Args:
            audio_data: Audio data as bytes
            ws: WebSocket connection
            is_interrupt: Whether this is a barge-in interruption
            is_indicator: Whether this is a listening indicator sound
        """
        try:
            # Ensure the audio data is valid
            if not audio_data or len(audio_data) == 0:
                logger.warning("Attempted to send empty audio data")
                return
                
            # Skip very small chunks unless they're for interruption or indicator
            if len(audio_data) < 320 and not (is_interrupt or is_indicator):
                logger.debug(f"Skipping very small audio chunk: {len(audio_data)} bytes")
                return
                
            # Check connection status
            if not self.connected:
                logger.warning("WebSocket connection is closed, cannot send audio")
                return
            
            # Special handling for indicator sounds
            if is_indicator:
                # Use a shorter, simplified sending process for indicators
                try:
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    message = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": audio_base64
                        }
                    }
                    ws.send(json.dumps(message))
                    logger.debug("Sent listening indicator sound")
                except Exception as e:
                    logger.error(f"Error sending indicator sound: {e}")
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
            if not is_interrupt and not self.audio_processor.preprocessor.agent_speaking:
                self.audio_processor.set_agent_speaking(True)
            
            # Send a few chunks of silence first if this is an interruption
            if is_interrupt:
                # Send 5 chunks (100ms) of silence to stop current audio
                silence_chunk = bytes([0x7f] * chunk_size)
                for _ in range(5):
                    try:
                        audio_base64 = base64.b64encode(silence_chunk).decode('utf-8')
                        message = {
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {
                                "payload": audio_base64
                            }
                        }
                        ws.send(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Error sending silence chunk: {e}")
                
                # Log the interruption
                logger.info("Sent silence to interrupt current speech")
                
                # Reset agent speaking flag immediately
                self.audio_processor.set_agent_speaking(False)
                return
            
            for i, chunk in enumerate(chunks):
                # Check if we should cancel speech (barge-in detected)
                if self.speech_cancellation_event.is_set():
                    logger.info(f"Cancelling speech output at chunk {i}/{len(chunks)} due to barge-in")
                    # Send a few silence frames to clear the audio buffer
                    silence_chunk = bytes([0x7f] * chunk_size)
                    try:
                        for _ in range(3):  # Send 3 chunks of silence
                            audio_base64 = base64.b64encode(silence_chunk).decode('utf-8')
                            message = {
                                "event": "media",
                                "streamSid": self.stream_sid,
                                "media": {
                                    "payload": audio_base64
                                }
                            }
                            ws.send(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Error sending silence on cancel: {e}")
                    
                    # Important: stop sending immediately after silence
                    break
                
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
                    if i % 5 == 0:  # Every 5 chunks (increased frequency)
                        await asyncio.sleep(0.005)  # 5ms delay
                    
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
            if not is_interrupt and self.audio_processor.preprocessor.agent_speaking:
                self.audio_processor.set_agent_speaking(False)
                
                # Update echo detection variables
                if hasattr(self, 'last_transcription') and self.last_transcription:
                    # Store the knowledge base response, not the transcription
                    response_text = ""
                    if self.conversation_manager and self.conversation_manager.history:
                        response_text = self.conversation_manager.history[-1].response
                    elif hasattr(self.pipeline, 'conversation_manager') and self.pipeline.conversation_manager.history:
                        response_text = self.pipeline.conversation_manager.history[-1].response
                    
                    # If we can't get the exact response, fall back to last_transcription
                    # This is still useful for echo detection
                    if not response_text:
                        response_text = self.last_transcription
                    
                    self.agent_last_response = response_text
                self.last_agent_speech_time = time.time()
            
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
            # Set state to responding
            self._set_conversation_state(ConversationState.RESPONDING)
            
            # Convert text to speech
            if hasattr(self.pipeline, 'tts_integration'):
                # Set agent speaking state
                self.audio_processor.set_agent_speaking(True)
                self.speech_cancellation_event.clear()
                
                speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                
                # Convert to mulaw
                mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                
                # Send audio
                await self._send_audio(mulaw_audio, ws)
                logger.info(f"Sent text response: '{text}'")
                
                # Reset agent speaking state
                self.audio_processor.set_agent_speaking(False)
                
                # Update last response time to add pause
                self.last_response_time = time.time()
                
                # After response completes, set back to listening
                self._set_conversation_state(ConversationState.LISTENING)
            else:
                logger.error("TTS integration not available")
                # If error, go back to listening
                self._set_conversation_state(ConversationState.LISTENING)
        except Exception as e:
            logger.error(f"Error sending text response: {e}", exc_info=True)
            # On error, go back to listening
            self._set_conversation_state(ConversationState.LISTENING)
    
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