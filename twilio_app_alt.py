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
import requests
import time
from flask import Flask, request, Response
from simple_websocket import Server, ConnectionClosed
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client

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

# Get Twilio credentials from environment
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Configure logging with more debug info
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)

# Global instances
twilio_handler = None
voice_ai_pipeline = None
base_url = None

async def initialize_system():
    """Initialize all system components."""
    global twilio_handler, voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent...")
    
    # Initialize Voice AI Agent
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        whisper_model_path='models/base.en',
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

@app.route('/', methods=['GET'])
def index():
    """Simple test endpoint."""
    return "Voice AI Agent is running!"

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls."""
    logger.info("Received incoming call request")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request form data: {request.form}")
    
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

@app.route('/voice/record', methods=['POST'])
def handle_recording():
    """Handle voice recording and process it through the AI pipeline."""
    logger.info("Received recording")
    logger.info(f"Recording data: {request.form}")
    
    # Get the recording data
    recording_sid = request.form.get('RecordingSid')
    call_sid = request.form.get('CallSid')
    
    # Create TwiML response
    response = VoiceResponse()
    
    try:
        # Download and process the recording
        if recording_sid and voice_ai_pipeline:
            # Wait a moment for the recording to be available
            time.sleep(2)
            
            # Use Twilio client to fetch the recording
            try:
                client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                
                # Fetch the recording
                recording = client.recordings(recording_sid).fetch()
                logger.info(f"Recording fetched. Media URL: {recording.media_url}")
                
                # Download the recording
                media_url = f'https://api.twilio.com{recording.media_url}'
                auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                recording_response = requests.get(media_url, auth=auth)
                
                if recording_response.status_code == 200:
                    # Save temporarily
                    temp_path = f'/tmp/{call_sid}.wav'
                    with open(temp_path, 'wb') as f:
                        f.write(recording_response.content)
                    
                    logger.info(f"Downloaded recording to {temp_path}")
                    
                    # Process through pipeline
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        voice_ai_pipeline.process_audio_file(temp_path)
                    )
                    loop.close()
                    
                    logger.info(f"Pipeline result: {result}")
                    
                    # Get the AI response
                    if result and 'response' in result:
                        response.say(result['response'], voice='alice')
                    else:
                        response.say("I'm sorry, I couldn't process that. Could you try again?", voice='alice')
                    
                    # Clean up
                    os.remove(temp_path)
                    logger.info(f"Removed temporary file {temp_path}")
                else:
                    logger.error(f"Failed to download recording. Status: {recording_response.status_code}")
                    response.say("I couldn't access your recording. Let me try again.", voice='alice')
                    response.say("Please speak after the beep.", voice='alice')
                    response.record(
                        action=f'{base_url}/voice/record',
                        method='POST',
                        timeout=10,
                        transcribe=False
                    )
                    return Response(str(response), mimetype='text/xml')
            except Exception as e:
                logger.error(f"Error using Twilio client: {e}", exc_info=True)
                response.say("I'm having trouble accessing your recording. Please try again.", voice='alice')
                response.say("Please speak after the beep.", voice='alice')
                response.record(
                    action=f'{base_url}/voice/record',
                    method='POST',
                    timeout=10,
                    transcribe=False
                )
                return Response(str(response), mimetype='text/xml')
        else:
            logger.error("No recording SID or pipeline not initialized")
            response.say("I'm having trouble processing your request. Please try again.", voice='alice')
        
        # Ask if they want to continue or end the call
        response.pause(length=1)
        response.say("Would you like to ask another question? Press 1 to continue or hang up to end the call.", voice='alice')
        
        # Gather digit input
        gather = response.gather(num_digits=1, action=f'{base_url}/voice/continue', method='POST')
        
    except Exception as e:
        logger.error(f"Error processing recording: {e}", exc_info=True)
        response.say("I encountered an error. Please try again.", voice='alice')
    
    return Response(str(response), mimetype='text/xml')

@app.route('/voice/continue', methods=['POST'])
def handle_continue():
    """Handle the continuation choice."""
    digits = request.form.get('Digits')
    response = VoiceResponse()
    
    if digits == '1':
        # Continue the conversation
        response.say("Please speak after the beep.", voice='alice')
        response.record(
            action=f'{base_url}/voice/record',
            method='POST',
            timeout=10,
            transcribe=False
        )
    else:
        # End the call
        response.say("Thank you for using Voice AI Agent. Goodbye!", voice='alice')
        response.hangup()
    
    return Response(str(response), mimetype='text/xml')

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
    logger.info(f"WebSocket connection attempt for call {call_sid}")
    logger.info(f"WebSocket environment: {request.environ}")
    
    if not twilio_handler or not voice_ai_pipeline:
        logger.error("System not initialized")
        return
    
    try:
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Send a ping message to test connection
        ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        
        # Create WebSocket handler
        ws_handler = WebSocketHandler(call_sid, voice_ai_pipeline)
        
        while True:
            try:
                message = ws.receive(timeout=30)  # Add timeout
                if message is None:
                    logger.warning(f"Received None message for call {call_sid}")
                    break
                
                logger.debug(f"Received message: {message[:100]}...")  # Log first 100 chars
                
                # Run async handler
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(ws_handler.handle_message(message, ws))
                loop.close()
                
            except ConnectionClosed:
                logger.info(f"WebSocket connection closed for call {call_sid}")
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket cleanup for call {call_sid}")
        try:
            if 'ws' in locals():
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