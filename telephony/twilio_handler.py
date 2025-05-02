"""
Twilio handler for voice calls with Google Cloud STT integration.
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

logger = logging.getLogger(__name__)

class TwilioHandler:
    """
    Handles Twilio voice call operations with Google Cloud STT integration.
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
    
    def handle_incoming_call(self, call_sid: str, from_number: str, to_number: str) -> str:
        """
        Handle incoming voice call with WebSocket streaming.
        
        Args:
            call_sid: Twilio call SID
            from_number: Caller phone number
            to_number: Called phone number
            
        Returns:
            TwiML response as string
        """
        logger.info(f"Incoming call from {from_number} to {to_number} (SID: {call_sid})")
        
        # Add call to manager
        self.call_manager.add_call(call_sid, from_number, to_number)
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Add initial greeting - shorter and clearer
        response.say("Hello, I'm your voice assistant. How can I help you today?", voice='alice')
        
        try:
            # Define WebSocket URL for streaming
            ws_url = f'{self.base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
            logger.info(f"Setting up WebSocket stream at: {ws_url}")
            
            # Use Connect and Stream for bidirectional audio streaming
            connect = Connect()
            stream = Stream(url=ws_url)
            connect.append(stream)
            response.append(connect)
            
            return str(response)
        except Exception as e:
            logger.error(f"Error setting up streaming: {e}")
            
            # Fallback to basic TwiML if streaming setup fails
            response = VoiceResponse()
            response.say("I'm sorry, but I'm having trouble connecting. Please try again later.", voice='alice')
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
    
    def place_outbound_call(
        self, 
        to_number: str, 
        from_number: Optional[str] = None,
        status_callback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place an outbound call with improved TwiML settings.
        
        Args:
            to_number: Destination phone number
            from_number: Caller ID (defaults to configured number)
            status_callback: URL for status callbacks
            
        Returns:
            Twilio call information
        """
        if not from_number:
            from_number = TWILIO_PHONE_NUMBER
        
        if not status_callback:
            status_callback = f"{self.base_url}/voice/status"
        
        # Set up TwiML for outbound call - using shorter, clearer messaging
        twiml = f"""
        <Response>
            <Say voice="alice">Hello! This is a call from the AI Voice Assistant.</Say>
            <Connect>
                <Stream url="{self.base_url.replace('https://', 'wss://')}/ws/stream/outbound" />
            </Connect>
        </Response>
        """
        
        try:
            # Place call using Twilio client
            call = self.client.calls.create(
                to=to_number,
                from_=from_number,
                twiml=twiml,
                status_callback=status_callback,
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                status_callback_method='POST'
            )
            
            logger.info(f"Placed outbound call to {to_number} (SID: {call.sid})")
            
            # Add to call manager
            self.call_manager.add_call(call.sid, from_number, to_number)
            
            return {
                "call_sid": call.sid,
                "status": call.status,
                "to": call.to,
                "from": call.from_,
                "direction": call.direction
            }
        except Exception as e:
            logger.error(f"Error placing outbound call: {e}")
            return {
                "error": str(e),
                "success": False
            }