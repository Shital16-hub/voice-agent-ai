"""
WebSocket handler for Twilio media streams.
"""
import json
import base64
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Callable, Awaitable

from telephony.audio_processor import AudioProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION

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
            elif event_type == 'start':
                await self._handle_start(data)
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
    
    async def _handle_start(self, data: Dict[str, Any]) -> None:
        """Handle stream start event."""
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        logger.info(f"Start data: {data}")
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event with audio data."""
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
            
            # Log for debugging
            if len(self.input_buffer) % 1000 == 0:  # Log every 1000 bytes
                logger.debug(f"Input buffer size: {len(self.input_buffer)} bytes")
            
            # Process when buffer is full enough
            if len(self.input_buffer) >= AUDIO_BUFFER_SIZE and not self.is_processing:
                logger.info(f"Processing buffer of size: {len(self.input_buffer)}")
                await self._process_audio(ws)
        except Exception as e:
            logger.error(f"Error processing media payload: {e}", exc_info=True)
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop event."""
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.conversation_active = False
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """Handle mark event for audio playback tracking."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    async def _process_audio(self, ws) -> None:
        """Process accumulated audio data through the pipeline."""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            # Convert buffer to PCM
            mulaw_bytes = bytes(self.input_buffer)
            pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
            
            # Clear input buffer
            self.input_buffer.clear()
            
            logger.debug(f"Processing audio chunk of size: {len(pcm_audio)}")
            
            # For now, just send a test response
            await self.send_test_response(ws, "I received your audio.")
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
        finally:
            self.is_processing = False
    
    async def send_test_response(self, ws, text: str) -> None:
        """Send a test text response."""
        try:
            # Convert text to speech using pipeline
            audio_data = await self.pipeline.tts_integration.text_to_speech(text)
            
            # Convert to mulaw
            mulaw_audio = self.audio_processor.pcm_to_mulaw(audio_data)
            
            # Send the audio
            await self._send_audio(mulaw_audio, ws)
            
        except Exception as e:
            logger.error(f"Error sending test response: {e}", exc_info=True)
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to Twilio."""
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