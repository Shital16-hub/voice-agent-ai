"""
WebSocket handler for Twilio media streams.
"""
import json
import base64
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Callable, Awaitable, Optional

from telephony.audio_processor import AudioProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_BUFFER_SIZE

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams.
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
        
        # Transcription tracker to avoid duplicate processing
        self.last_transcription = ""
        self.last_response_time = 0
        self.processing_lock = asyncio.Lock()
        self.keep_alive_task = None
        
        # Create an event to signal when we should stop processing
        self.stop_event = asyncio.Event()
        
        logger.info(f"WebSocketHandler initialized for call {call_sid}")
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            logger.debug(f"Received WebSocket event: {event_type}")
            
            if event_type == 'connected':
                logger.info(f"WebSocket connected for call {self.call_sid}")
                logger.info(f"Connected data: {data}")
                
                # Start keep-alive task
                if not self.keep_alive_task:
                    self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
                    
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
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
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
        self.conversation_active = True
        self.stop_event.clear()
        
        # Send a welcome message
        await self.send_text_response("I'm listening. How can I help you today?", ws)
    
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
        self.stop_event.set()
        
        # Cancel keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Process any remaining audio
        if len(self.input_buffer) > 0:
            try:
                # Try to process any remaining audio in the buffer
                logger.info(f"Processing final audio chunk of size: {len(self.input_buffer)}")
                
                # Convert buffer to PCM
                mulaw_bytes = bytes(self.input_buffer)
                pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
                self.input_buffer.clear()
                
                # Create transcription result directly
                if hasattr(self.pipeline, 'speech_recognizer'):
                    # Ensure we're starting fresh
                    self.pipeline.speech_recognizer.start_streaming()
                    
                    # Process the accumulated audio
                    await self.pipeline.speech_recognizer.process_audio_chunk(pcm_audio)
                    
                    # Get the final transcription
                    transcription, duration = await self.pipeline.speech_recognizer.stop_streaming()
                    logger.info(f"Final transcription: {transcription}")
                    
                    # If we got a valid transcription, try to respond
                    if transcription and transcription.strip() and hasattr(self.pipeline, 'query_engine'):
                        query_result = await self.pipeline.query_engine.query(transcription)
                        response = query_result.get("response", "")
                        logger.info(f"Final response: {response}")
            except Exception as e:
                logger.error(f"Error processing final audio: {e}", exc_info=True)
    
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
        Process accumulated audio data through the pipeline.
        
        Args:
            ws: WebSocket connection
        """
        try:
            # Convert buffer to PCM
            mulaw_bytes = bytes(self.input_buffer)
            pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
            
            logger.debug(f"Processing audio chunk of size: {len(pcm_audio)} samples")
            
            # Check if audio contains speech before processing
            if not self._contains_speech(pcm_audio):
                logger.debug("Audio doesn't contain speech, skipping processing")
                # Still clear some of the buffer to avoid accumulating silence
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
                
            # Only process if we have enough audio (at least 1.5 seconds worth)
            sr = getattr(self.pipeline.speech_recognizer, 'sample_rate', 16000)
            min_samples = int(sr * 1.5)  # 1.5 seconds of audio
            
            if len(pcm_audio) < min_samples:
                logger.debug(f"Audio chunk too short ({len(pcm_audio)} samples), accumulating more audio")
                return
            
            # Set up transcription callback for streaming results
            async def transcription_callback(result):
                if result and result.text and result.text.strip():
                    text = result.text.strip()
                    logger.info(f"Interim transcription: {text}")
            
            # Start STT streaming if not already active
            if hasattr(self.pipeline, 'speech_recognizer'):
                # Reset any existing streaming session
                if hasattr(self.pipeline.speech_recognizer, 'is_streaming') and self.pipeline.speech_recognizer.is_streaming:
                    await self.pipeline.speech_recognizer.stop_streaming()
                
                # Start a new streaming session
                self.pipeline.speech_recognizer.start_streaming()
                
                # Process audio chunk
                await self.pipeline.speech_recognizer.process_audio_chunk(
                    audio_chunk=pcm_audio,
                    callback=transcription_callback
                )
                
                # Check for complete transcription
                transcription, duration = await self.pipeline.speech_recognizer.stop_streaming()
                
                # Only clear the buffer and process if we got a valid transcription
                if transcription and transcription.strip():
                    logger.info(f"Complete transcription: {transcription}")
                    
                    # Now clear the input buffer since we have a valid transcription
                    self.input_buffer.clear()
                    
                    # Process through knowledge base
                    try:
                        if hasattr(self.pipeline, 'query_engine'):
                            query_result = await self.pipeline.query_engine.query(transcription)
                            response = query_result.get("response", "")
                            
                            logger.info(f"Generated response: {response}")
                            
                            # Convert to speech
                            if response and hasattr(self.pipeline, 'tts_integration'):
                                speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                
                                # Convert to mulaw for Twilio
                                mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                                
                                # Send back to Twilio
                                logger.info(f"Sending audio response ({len(mulaw_audio)} bytes)")
                                await self._send_audio(mulaw_audio, ws)
                                
                                # Update state
                                self.last_transcription = transcription
                                self.last_response_time = asyncio.get_event_loop().time()
                    except Exception as e:
                        logger.error(f"Error processing through knowledge base: {e}", exc_info=True)
                        
                        # Try to send a fallback response
                        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
                        try:
                            fallback_audio = await self.pipeline.tts_integration.text_to_speech(fallback_message)
                            mulaw_fallback = self.audio_processor.pcm_to_mulaw(fallback_audio)
                            await self._send_audio(mulaw_fallback, ws)
                        except Exception as e2:
                            logger.error(f"Failed to send fallback response: {e2}")
                else:
                    # If no transcription, reduce buffer size but keep some for context
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    logger.debug(f"No valid transcription, reduced buffer to {len(self.input_buffer)} bytes")
            else:
                logger.error("Speech recognizer not available in pipeline")
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
    
    def _contains_speech(self, audio_data: np.ndarray, threshold: float = 0.005) -> bool:
        """
        Check if audio contains any speech.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Energy threshold for speech detection
            
        Returns:
            True if audio contains potential speech
        """
        # Calculate energy in different frequency bands
        if len(audio_data) < 100:
            return False
            
        # Simple energy check
        energy = np.mean(np.abs(audio_data))
        has_energy = energy > threshold
        
        # Also check for variance (speech has more variance than background noise)
        variance = np.var(audio_data)
        has_variance = variance > (threshold * 0.1)
        
        logger.debug(f"Audio energy: {energy:.6f}, variance: {variance:.6f}, contains speech: {has_energy and has_variance}")
        
        return has_energy and has_variance
    
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
            
            # Encode audio to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
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
            logger.debug(f"Sent audio data: {len(audio_data)} bytes")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=True)
    
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
                await asyncio.sleep(20)  # Send every 20 seconds
                await self._send_keep_alive(ws)
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")
    
    async def _send_keep_alive(self, ws) -> None:
        """Send a keep-alive message to maintain the WebSocket connection."""
        try:
            if self.stream_sid:
                message = {
                    "event": "ping",
                    "streamSid": self.stream_sid
                }
                ws.send(json.dumps(message))
                logger.debug("Sent keep-alive ping")
        except Exception as e:
            logger.error(f"Error sending keep-alive: {e}")