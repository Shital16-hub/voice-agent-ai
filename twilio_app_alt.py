
#!/usr/bin/env python3
"""
Alternative Twilio application using simple-websocket.
"""
import os
import sys
import asyncio
import logging
import json
import base64
from flask import Flask, request, Response
from simple_websocket import Server, ConnectionClosed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telephony.twilio_handler import TwilioHandler
from telephony.websocket_handler import WebSocketHandler
from telephony.config import HOST, PORT, DEBUG, LOG_LEVEL, LOG_FORMAT
from voice_ai_agent import VoiceAIAgent
from integration.tts_integration import TTSIntegration
from integration.pipeline import VoiceAIAgentPipeline

# Configure logging with more debug info
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)

# Global instances
twilio_handler = None
voice_ai_pipeline = None

async def initialize_system():
    """Initialize all system components."""
    global twilio_handler, voice_ai_pipeline
    
    logger.info("Initializing Voice AI Agent...")
    
    # Initialize Voice AI Agent
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        whisper_model_path='tiny.en',
        llm_temperature=0.7
    )
    await agent.init()
    
    # Initialize TTS integration
    tts = TTSIntegration()
    await tts.init()
    
    # Create pipeline
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    # Get base URL from environment
    base_url = os.getenv('BASE_URL')
    if not base_url:
        logger.error("BASE_URL not set in environment")
        raise ValueError("BASE_URL must be set")
    
    logger.info(f"Using BASE_URL: {base_url}")
    
    # Initialize Twilio handler
    twilio_handler = TwilioHandler(voice_ai_pipeline, base_url)
    await twilio_handler.start()
    
    logger.info("System initialized successfully")

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls."""
    logger.info("Received incoming call request")
    
    if not twilio_handler:
        logger.error("System not initialized")
        fallback_twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>System is not initialized. Please try again later.</Say>
</Response>'''
        return Response(fallback_twiml, mimetype='text/xml')
    
    # Get call parameters
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    call_sid = request.form.get('CallSid')
    
    logger.info(f"Incoming call - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    try:
        # Generate TwiML response from twilio_handler
        twiml = twilio_handler.handle_incoming_call(from_number, to_number, call_sid)
        logger.info(f"Generated TwiML: {twiml}")
        
        return Response(twiml, mimetype='text/xml')
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        fallback_twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>An error occurred. Please try again later.</Say>
</Response>'''
        return Response(fallback_twiml, mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks."""
    logger.info("Received status callback")
    logger.info(f"Status data: {request.form}")
    
    if not twilio_handler:
        logger.error("System not initialized")
        return Response('', status=204)
    
    # Get status parameters
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    try:
        # Handle status update
        twilio_handler.handle_status_callback(call_sid, call_status)
        return Response('', status=204)
    except Exception as e:
        logger.error(f"Error handling status callback: {e}", exc_info=True)
        return Response('', status=204)

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream."""
    if not twilio_handler or not voice_ai_pipeline:
        logger.error("System not initialized")
        return
    
    ws = Server.accept(request.environ)
    logger.info(f"WebSocket connection established for call {call_sid}")
    
    # Create WebSocket handler
    ws_handler = WebSocketHandler(call_sid, voice_ai_pipeline)
    
    try:
        while True:
            try:
                message = ws.receive()
                if message is None:
                    break
                
                # Log the type of message received
                try:
                    msg_data = json.loads(message)
                    msg_type = msg_data.get('event', 'unknown')
                    logger.debug(f"Received WebSocket message type: {msg_type}")
                except:
                    logger.debug("Received non-JSON WebSocket message")
                
                # Run async handler in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(ws_handler.handle_message(message, ws))
                loop.close()
                
            except ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
    
    finally:
        logger.info(f"WebSocket closed for call {call_sid}")
        try:
            ws.close()
        except:
            pass

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if not twilio_handler or not voice_ai_pipeline:
        return Response(json.dumps({"status": "unhealthy"}), 
                       mimetype='application/json', 
                       status=503)
    
    return Response(json.dumps({"status": "healthy"}), 
                   mimetype='application/json')

if __name__ == '__main__':
    # Initialize the system
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialize_system())
    
    # Run the server
    logger.info(f"Server starting on {HOST}:{PORT}")
    logger.info(f"BASE_URL: {os.getenv('BASE_URL')}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
