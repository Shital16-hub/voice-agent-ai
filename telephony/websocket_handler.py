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
            elif event_type == 'start':
                await self._handle_start(data)
            elif event_type == 'media':
                await self._handle_media(data, ws)
            elif event_type == 'stop':
                await self._handle_stop(data)
            elif event_type == 'mark':
                await self._handle_mark(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _handle_start(self, data: Dict[str, Any]) -> None:
        """Handle stream start event."""
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event with audio data."""
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
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
            pcm_audio = self.audio_processor.mulaw_to_pcm(self.input_buffer)
            
            # Clear input buffer
            self.input_buffer.clear()
            
            logger.debug(f"Processing audio chunk of size: {len(pcm_audio)}")
            
            # Check for silence
            if self.audio_processor.detect_silence(pcm_audio, SILENCE_THRESHOLD):
                if not self.silence_start_time:
                    self.silence_start_time = asyncio.get_event_loop().time()
                    logger.debug("Silence detected - starting timer")
                elif asyncio.get_event_loop().time() - self.silence_start_time > SILENCE_DURATION:
                    # Detected end of utterance
                    if self.is_speaking:
                        logger.info("Detected end of utterance")
                        await self._handle_end_of_utterance(ws)
                        self.is_speaking = False
            else:
                if not self.is_speaking:
                    logger.info("Detected speech start")
                self.is_speaking = True
                self.silence_start_time = None
            
            # Process through pipeline if speaking
            if self.is_speaking:
                logger.info("Processing speech through pipeline")
                await self._pipeline_process(pcm_audio, ws)
        
        finally:
            self.is_processing = False
    
    async def _handle_end_of_utterance(self, ws) -> None:
        """Handle end of user utterance."""
        logger.info("End of utterance detected")
        # Add any processing needed for end of utterance
    
    async def _pipeline_process(self, audio_data: np.ndarray, ws) -> None:
        """Process audio through the Voice AI pipeline."""
        try:
            logger.info(f"Processing audio data of length: {len(audio_data)}")
            
            # Create audio callback for TTS output
            async def audio_callback(tts_audio: bytes):
                # Convert TTS output to mulaw
                mulaw_audio = self.audio_processor.pcm_to_mulaw(tts_audio)
                
                # Send to Twilio
                await self._send_audio(mulaw_audio, ws)
                logger.info(f"Sent {len(mulaw_audio)} bytes of audio response to Twilio")
            
            # Check if we have the correct pipeline method
            if hasattr(self.pipeline, 'process_audio_streaming'):
                # Process through pipeline
                result = await self.pipeline.process_audio_streaming(
                    audio_data,
                    audio_callback=audio_callback
                )
                logger.info(f"Pipeline streaming result: {result}")
            else:
                # Fallback to regular processing
                logger.warning("Using fallback audio processing")
                result = await self.pipeline.process_audio_data(
                    audio_data,
                    speech_output_path=None
                )
                
                logger.info(f"Pipeline result: {result}")
                
                # If we got a response, send it back
                if result and result.get('speech_audio'):
                    mulaw_audio = self.audio_processor.pcm_to_mulaw(result['speech_audio'])
                    await self._send_audio(mulaw_audio, ws)
                    logger.info(f"Sent {len(mulaw_audio)} bytes of audio response")
        
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}", exc_info=True)
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to Twilio."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
    
    async def send_clear_message(self, ws) -> None:
        """Send clear message to stop any ongoing audio playback."""
        try:
            message = {
                "event": "clear",
                "streamSid": self.stream_sid
            }
            ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending clear message: {e}")
    
    async def send_mark(self, ws, name: str) -> None:
        """Send mark message for tracking audio playback."""
        try:
            message = {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {
                    "name": name
                }
            }
            ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending mark: {e}")