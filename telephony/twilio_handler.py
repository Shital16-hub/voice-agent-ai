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
        response.say("Welcome to the Voice AI Agent. I'm here to help. Please speak after the beep.", voice='alice')
        response.pause(length=1)
        
        # Try the record approach as a fallback
        response.record(
            action=f'{self.base_url}/voice/record',
            method='POST',
            timeout=10,
            transcribe=False
        )
        
        return str(response)
    
    def handle_incoming_call_with_stream(self, from_number: str, to_number: str, call_sid: str) -> str:
        """
        Handle incoming voice call with WebSocket stream.
        This is the original method with streaming.
        """
        logger.info(f"Incoming call from {from_number} to {to_number} (SID: {call_sid})")
        
        # Add call to manager
        self.call_manager.add_call(call_sid, from_number, to_number)
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Add initial greeting
        response.say("Welcome to the Voice AI Agent. I'm here to help.", voice='alice')
        response.pause(length=1)
        
        # Try a direct stream connection
        ws_url = f'wss://0ulckcn58gni70-5000.proxy.runpod.net/ws/stream/{call_sid}'
        
        # Use the connect.stream() approach
        connect = Connect()
        stream = Stream(url=ws_url)
        connect.append(stream)
        response.append(connect)
        
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