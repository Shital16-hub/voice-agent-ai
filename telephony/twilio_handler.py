"""
Main Twilio handler for voice calls.
"""
import logging
from typing import Optional, Dict, Any
from twilio.twiml.voice_response import VoiceResponse, Connect, Start, Stream
from twilio.rest import Client

from telephony.config import (
    TWILIO_ACCOUNT_SID, 
    TWILIO_AUTH_TOKEN, 
    TWILIO_PHONE_NUMBER,
    MAX_CALL_DURATION
)
from telephony.call_manager import CallManager
from telephony.websocket_handler import WebSocketHandler

logger = logging.getLogger(__name__)

class TwilioHandler:
    """
    Handles Twilio voice call operations.
    """
    
    def __init__(self, pipeline, base_url: str):
        """
        Initialize Twilio handler.
        
        Args:
            pipeline: Voice AI pipeline instance
            base_url: Base URL for webhooks (e.g., https://your-domain.com)
        """
        self.pipeline = pipeline
        self.base_url = base_url.rstrip('/')
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.call_manager = CallManager()
    
    async def start(self):
        """Start the Twilio handler."""
        await self.call_manager.start()
        logger.info("Twilio handler started")
    
    async def stop(self):
        """Stop the Twilio handler."""
        await self.call_manager.stop()
        logger.info("Twilio handler stopped")
    
    def handle_incoming_call(self, from_number: str, to_number: str, call_sid: str) -> str:
        """
        Handle incoming voice call.
        
        Args:
            from_number: Caller phone number
            to_number: Called phone number
            call_sid: Twilio call SID
            
        Returns:
            TwiML response as string
        """
        logger.info(f"Incoming call from {from_number} to {to_number} (SID: {call_sid})")
        
        # Add call to manager
        self.call_manager.add_call(call_sid, from_number, to_number)
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Add initial greeting
        response.say("Welcome to the Voice AI Agent. I'm here to help. Please speak after the beep.", 
                    voice='alice')
        response.pause(length=1)
        
        # Start media stream
        start = Start()
        stream = Stream(url=f'wss://{self.base_url.replace("https://", "").replace("http://", "")}/ws/stream/{call_sid}')
        start.append(stream)
        response.append(start)
        
        # Keep the call alive
        response.pause(length=MAX_CALL_DURATION)
        
        return str(response)
    
    def handle_status_callback(self, call_sid: str, call_status: str) -> None:
        """
        Handle call status callback.
        
        Args:
            call_sid: Twilio call SID
            call_status: Call status
        """
        logger.info(f"Call {call_sid} status: {call_status}")
        
        # Update call status
        self.call_manager.update_call_status(call_sid, call_status)
        
        # Remove call if completed or failed
        if call_status in ['completed', 'failed', 'busy', 'no-answer']:
            self.call_manager.remove_call(call_sid)
    
    async def handle_websocket_connection(self, call_sid: str, ws) -> None:
        """
        Handle WebSocket connection for media stream.
        
        Args:
            call_sid: Twilio call SID
            ws: WebSocket connection
        """
        try:
            # Create WebSocket handler
            ws_handler = WebSocketHandler(call_sid, self.pipeline)
            
            # Process messages
            async for message in ws:
                if message is None:
                    break
                await ws_handler.handle_message(message, ws)
                
        except Exception as e:
            logger.error(f"WebSocket error for call {call_sid}: {e}")
        finally:
            logger.info(f"WebSocket closed for call {call_sid}")
    
    def make_call(self, to_number: str, from_number: Optional[str] = None) -> str:
        """
        Make an outbound call.
        
        Args:
            to_number: Phone number to call
            from_number: Phone number to call from (defaults to Twilio number)
            
        Returns:
            Call SID
        """
        from_number = from_number or TWILIO_PHONE_NUMBER
        
        try:
            call = self.client.calls.create(
                url=f'{self.base_url}/voice/outgoing',
                to=to_number,
                from_=from_number,
                status_callback=f'{self.base_url}/voice/status',
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                status_callback_method='POST'
            )
            
            logger.info(f"Initiated call to {to_number} (SID: {call.sid})")
            return call.sid
            
        except Exception as e:
            logger.error(f"Error making call to {to_number}: {e}")
            raise
    
    def handle_outgoing_call(self) -> str:
        """
        Handle outgoing call TwiML.
        
        Returns:
            TwiML response as string
        """
        response = VoiceResponse()
        
        # Add greeting for outgoing calls
        response.say("Hello, this is the Voice AI Agent calling. How can I help you today?", 
                    voice='alice')
        response.pause(length=1)
        
        # Start media stream (same as incoming)
        start = Start()
        stream = Stream(url=f'wss://{self.base_url.replace("https://", "").replace("http://", "")}/ws/stream')
        start.append(stream)
        response.append(start)
        
        # Keep the call alive
        response.pause(length=MAX_CALL_DURATION)
        
        return str(response)
    
    def get_call_stats(self) -> Dict[str, Any]:
        """Get call statistics."""
        return self.call_manager.get_call_stats()
    
    def get_call_info(self, call_sid: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific call."""
        return self.call_manager.get_call(call_sid)