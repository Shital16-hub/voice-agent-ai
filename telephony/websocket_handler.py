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
from typing import Dict, Any, Callable, Awaitable, Optional, List, Union
from collections import deque

from telephony.audio_preprocessor import AudioPreprocessor, SpeechState
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

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams with enhanced speech/noise discrimination.
    Integrates the new AudioPreprocessor via the updated AudioProcessor.
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
        self.audio_processor = pipeline.speech_recognizer.audio_processor if hasattr(pipeline.speech_recognizer, 'audio_processor') else None
        
        # Initialize audio processor if not already present
        if not self.audio_processor:
            from telephony.audio_processor import AudioProcessor
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
        
        # Conversation flow management with increased pause
        self.pause_after_response = 4.0  # Increased from 3.0 seconds - wait 4 seconds after response
        self.min_words_for_valid_query = 3  # Increased from 2 - minimum 3 words for a valid query
        
        # Create an event to signal when we should stop processing
        self.stop_event = asyncio.Event()
        
        # Barge-in handling
        self.speech_cancellation_event = asyncio.Event()
        self.speech_cancellation_event.clear()
        
        # New: Add startup delay before processing audio (prevents false triggers at start)
        self.startup_time = time.time()
        self.startup_delay_sec = 2.0  # 2 second delay before processing audio
        
        # New: Track consecutive silent frames for better silence detection
        self.consecutive_silent_frames = 0
        self.min_silent_frames_for_silence = 10  # Increased from 5 - minimum 10 frames to be considered silence
        
        # New: Add a flag to prevent multiple barge-in detections in quick succession
        self.last_barge_in_time = 0
        self.barge_in_cooldown_sec = 1.5  # 1.5 second cooldown between barge-ins
        
        # Track chunks processed since last processing 
        self.chunks_since_last_process = 0
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with enhanced audio preprocessing")
    
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
        
        # Reset the audio processor
        self.audio_processor.reset()
        
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
            if self.audio_processor.preprocessor.agent_speaking:
                audio_array = self.audio_processor.mulaw_to_pcm(audio_data)
                # Ensure float32 format
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                    
                if self.audio_processor.check_for_barge_in(audio_array):
                    # Enough time since last barge-in?
                    current_time = time.time()
                    if current_time - self.last_barge_in_time > self.barge_in_cooldown_sec:
                        # Set the barge-in timestamp
                        self.last_barge_in_time = current_time
                        
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
                (self.audio_processor.preprocessor.speech_state == SpeechState.CONFIRMED_SPEECH or 
                 self.chunks_since_last_process >= 5)):  # Only process every 5 chunks if no speech
                
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
                
                # Ensure float32 format
                if pcm_audio.dtype != np.float32:
                    pcm_audio = pcm_audio.astype(np.float32)
                
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                # Clear part of buffer and try again next time
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
                
            # Add checks for audio quality
            if len(pcm_audio) < 1000:  # Very small audio chunk
                logger.warning(f"Audio chunk too small: {len(pcm_audio)} samples")
                return
            
            # Use enhanced preprocessor to check if audio contains actual speech
            contains_speech = self.audio_processor.contains_speech(pcm_audio)
            if not contains_speech:
                logger.info("No speech detected in audio buffer, skipping processing")
                
                # Increment silent frame counter
                self.consecutive_silent_frames += 1
                
                # Only reduce buffer if we've confirmed it's truly silence
                # (multiple consecutive silent frames)
                if self.consecutive_silent_frames >= self.min_silent_frames_for_silence:
                    # Reduce buffer size but keep some context
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    logger.debug(f"Confirmed silence, reduced buffer to {len(self.input_buffer)} bytes")
                    self.consecutive_silent_frames = 0  # Reset counter
                
                return
            
            # If we get here, we have speech - reset silent frame counter
            self.consecutive_silent_frames = 0
            
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
                    # Make sure streaming is started before processing
                    if hasattr(self.pipeline.speech_recognizer, 'start_streaming'):
                        # Check if it's an async method
                        if asyncio.iscoroutinefunction(self.pipeline.speech_recognizer.start_streaming):
                            # Make sure streaming is started
                            if not getattr(self.pipeline.speech_recognizer, 'is_streaming', False):
                                await self.pipeline.speech_recognizer.start_streaming()
                                # Add a small delay to ensure streaming is fully started
                                await asyncio.sleep(0.1)
                        else:
                            # Handle synchronous start_streaming
                            if not getattr(self.pipeline.speech_recognizer, 'is_streaming', False):
                                self.pipeline.speech_recognizer.start_streaming()
                                # Add a small delay to ensure streaming is fully started
                                await asyncio.sleep(0.1)
                
                # Determine what type of STT we're using by checking attributes and methods
                using_whisper = False
                using_deepgram = False
                
                if hasattr(self.pipeline, 'speech_recognizer'):
                    # Check for Whisper-specific attributes
                    if hasattr(self.pipeline.speech_recognizer, 'model'):
                        using_whisper = True
                    # Check for Deepgram-specific attributes
                    elif hasattr(self.pipeline.speech_recognizer, 'api_key'):
                        using_deepgram = True
                
                # For Deepgram, convert to bytes format
                audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
                    
                if using_whisper:
                    # Process with Whisper (expects numpy array)
                    await self.pipeline.speech_recognizer.process_audio_chunk(
                        audio_chunk=pcm_audio,  # Pass numpy array for Whisper
                        callback=transcription_callback
                    )
                    logger.debug("Processed audio with Whisper STT")
                elif using_deepgram:
                    # For Deepgram, ensure we're streaming first by checking and waiting
                    streaming_started = getattr(self.pipeline.speech_recognizer, 'streaming_started', None)
                    if streaming_started and hasattr(streaming_started, 'wait'):
                        try:
                            # Wait for streaming to be properly initialized
                            await asyncio.wait_for(streaming_started.wait(), timeout=2.0)
                        except asyncio.TimeoutError:
                            logger.warning("Timeout waiting for Deepgram streaming to initialize")
                            # Try restarting streaming
                            await self.pipeline.speech_recognizer.start_streaming()
                            # Wait again, but shorter timeout
                            try:
                                await asyncio.wait_for(streaming_started.wait(), timeout=1.0)
                            except:
                                pass
                    
                    # Process with Deepgram (expects bytes)
                    await self.pipeline.speech_recognizer.process_audio_chunk(
                        audio_chunk=audio_bytes,  # Use bytes for Deepgram
                        callback=transcription_callback
                    )
                    logger.debug("Processed audio with Deepgram STT")
                else:
                    # Try both formats if we can't determine the type
                    try:
                        # First try with numpy array
                        await self.pipeline.speech_recognizer.process_audio_chunk(
                            audio_chunk=pcm_audio,
                            callback=transcription_callback
                        )
                    except AttributeError:
                        # If that fails, try with bytes
                        await self.pipeline.speech_recognizer.process_audio_chunk(
                            audio_chunk=audio_bytes,
                            callback=transcription_callback
                        )
                
                # Get final transcription
                final_transcription = ""
                if hasattr(self.pipeline.speech_recognizer, 'stop_streaming'):
                    # Check if stop_streaming is async
                    try:
                        if asyncio.iscoroutinefunction(self.pipeline.speech_recognizer.stop_streaming):
                            result = await self.pipeline.speech_recognizer.stop_streaming()
                            # Check if result is None or a tuple
                            if result is not None:
                                # Handle as tuple
                                if isinstance(result, tuple) and len(result) >= 1:
                                    final_transcription, _ = result
                                else:
                                    # Handle as single value
                                    final_transcription = str(result)
                            else:
                                # Deepgram error case - result is None
                                final_transcription = ""
                        else:
                            # Handle synchronous stop_streaming
                            result = self.pipeline.speech_recognizer.stop_streaming()
                            # Similar unpacking with safety checks
                            if result is not None:
                                if isinstance(result, tuple) and len(result) >= 1:
                                    final_transcription, _ = result
                                else:
                                    final_transcription = str(result)
                            else:
                                final_transcription = ""
                    except Exception as e:
                        logger.error(f"Error getting final transcription: {e}")
                        final_transcription = ""
                    
                    # Add to results if not empty
                    if final_transcription:
                        # Create a simple result object to unify handling
                        class SimpleResult:
                            def __init__(self, text):
                                self.text = text
                                self.is_final = True
                                self.confidence = 1.0
                                
                        transcription_results.append(SimpleResult(final_transcription))
                
                # Find best result (highest confidence or longest text)
                best_result = None
                if transcription_results:
                    # First try to get final results with confidence
                    final_results = [r for r in transcription_results if 
                                    getattr(r, 'is_final', False) and 
                                    hasattr(r, 'text') and r.text]
                    
                    if final_results:
                        # Use highest confidence if available
                        if hasattr(final_results[0], 'confidence'):
                            best_result = max(final_results, key=lambda r: getattr(r, 'confidence', 0))
                        else:
                            # Otherwise use longest text
                            best_result = max(final_results, key=lambda r: len(getattr(r, 'text', '')))
                    else:
                        # No final results, use any result with text
                        text_results = [r for r in transcription_results if hasattr(r, 'text') and r.text]
                        if text_results:
                            best_result = max(text_results, key=lambda r: len(getattr(r, 'text', '')))
                
                # Get transcription from best result
                if best_result and hasattr(best_result, 'text'):
                    transcription = best_result.text
                else:
                    transcription = ""
                
                # Log before cleanup for debugging
                logger.info(f"RAW TRANSCRIPTION: '{transcription}'")
                
                # Clean up transcription
                transcription = self.cleanup_transcription(transcription)
                logger.info(f"CLEANED TRANSCRIPTION: '{transcription}'")
                
                # Check for noise keywords - skip knowledge base query for noise
                noise_keywords = ["click", "keyboard", "tongue", "scoff", "static", "*", "(", ")"]
                if any(keyword in transcription.lower() for keyword in noise_keywords):
                    logger.info(f"Detected noise keywords in transcription, skipping: '{transcription}'")
                    # Clear part of buffer and continue
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    return
                
                # Check for empty or short transcriptions
                if not transcription or len(transcription.split()) < self.min_words_for_valid_query:
                    logger.info(f"Transcription too short, skipping: '{transcription}'")
                    # Clear part of buffer and continue
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    return
                
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
                                    self.audio_processor.set_agent_speaking(True)
                                    self.speech_cancellation_event.clear()
                                    
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
                                        fallback_audio = bytes([0x7f] * silence_size)  # Simple audio
                                        await self._send_audio(fallback_audio, ws)
                                
                                finally:
                                    # Reset agent speaking state
                                    self.audio_processor.set_agent_speaking(False)
                                
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
                
                # Restart STT streaming session for next input
                if hasattr(self.pipeline, 'speech_recognizer'):
                    # Check if start_streaming is async
                    if asyncio.iscoroutinefunction(self.pipeline.speech_recognizer.start_streaming):
                        await self.pipeline.speech_recognizer.start_streaming()
                        # Add a small delay to ensure streaming is fully started
                        await asyncio.sleep(0.1)
                    else:
                        # Handle synchronous start_streaming
                        self.pipeline.speech_recognizer.start_streaming()
                        # Add a small delay to ensure streaming is fully started
                        await asyncio.sleep(0.1)
                    logger.info("Reset STT streaming session for next input")
            
            except Exception as e:
                logger.error(f"Error during STT processing: {e}", exc_info=True)
                # If error, clear part of buffer and continue
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                
                # Reset STT if needed
                if hasattr(self.pipeline, 'speech_recognizer'):
                    try:
                        # Try to reset STT with error handling for sync/async methods
                        if hasattr(self.pipeline.speech_recognizer, 'stop_streaming'):
                            try:
                                if asyncio.iscoroutinefunction(self.pipeline.speech_recognizer.stop_streaming):
                                    await self.pipeline.speech_recognizer.stop_streaming()
                                else:
                                    self.pipeline.speech_recognizer.stop_streaming()
                            except Exception:
                                pass
                        
                        await asyncio.sleep(0.2)  # Add delay between stop and start
                        
                        if hasattr(self.pipeline.speech_recognizer, 'start_streaming'):
                            if asyncio.iscoroutinefunction(self.pipeline.speech_recognizer.start_streaming):
                                await self.pipeline.speech_recognizer.start_streaming()
                                # Add a small delay to ensure streaming is fully started
                                await asyncio.sleep(0.1)
                            else:
                                self.pipeline.speech_recognizer.start_streaming()
                                # Add a small delay to ensure streaming is fully started
                                await asyncio.sleep(0.1)
                            logger.info("Reset STT after error")
                    except Exception as reset_error:
                        logger.error(f"Error resetting STT: {reset_error}")
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
    
    async def _send_audio(self, audio_data: bytes, ws, is_interrupt: bool = False) -> None:
        """
        Send audio data to Twilio with improved barge-in support.
        
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
            if not is_interrupt and not self.audio_processor.preprocessor.agent_speaking:
                self.audio_processor.set_agent_speaking(True)
            
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
            if not is_interrupt and self.audio_processor.preprocessor.agent_speaking:
                self.audio_processor.set_agent_speaking(False)
            
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