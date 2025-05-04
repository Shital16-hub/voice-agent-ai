#!/usr/bin/env python3
"""
Enhanced Twilio application using WebSocket streaming with improved noise handling.
"""
import os
import sys
import asyncio
import logging
import json
import base64
import requests
import time
import threading
from flask import Flask, request, Response
from simple_websocket import Server, ConnectionClosed
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client

# Load environment variables
load_dotenv()

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telephony.twilio_handler import TwilioHandler
from telephony.websocket_handler import WebSocketHandler
from telephony.config import HOST, PORT, DEBUG, LOG_LEVEL, LOG_FORMAT
from telephony.config import STT_INITIAL_PROMPT, STT_NO_CONTEXT, STT_TEMPERATURE, STT_PRESET
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

# Improved call event loop management
call_event_loops = {}
call_event_loops_lock = threading.Lock()

def get_call_event_loop(call_sid):
    """Get or create event loop for a call with thread safety."""
    with call_event_loops_lock:
        if call_sid not in call_event_loops:
            # Create new loop and related objects
            loop = asyncio.new_event_loop()
            terminate_flag = threading.Event()
            call_event_loops[call_sid] = {
                'loop': loop,
                'terminate_flag': terminate_flag,
                'handler': None,  # Will be set later
                'thread': None    # Will be set later
            }
        return call_event_loops[call_sid]

def remove_call_event_loop(call_sid):
    """Safely remove call event loop."""
    with call_event_loops_lock:
        if call_sid in call_event_loops:
            info = call_event_loops[call_sid]
            # Signal termination
            if 'terminate_flag' in info:
                info['terminate_flag'].set()
            # Remove from dictionary
            del call_event_loops[call_sid]
            return True
        return False

# Add to WebSocketHandler class
async def _ensure_google_session(self):
    """Ensure the Google Cloud STT session is active and properly initialized."""
    try:
        if hasattr(self.pipeline, 'speech_recognizer'):
            if not self.google_session_active:
                # Start a new session
                await self.pipeline.speech_recognizer.start_streaming()
                self.google_session_active = True
                logger.info("Started new Google Cloud STT session")
            return True
        else:
            logger.error("Speech recognizer not available in pipeline")
            return False
    except Exception as e:
        logger.error(f"Error ensuring Google STT session: {e}")
        # Try to reset session
        try:
            await self.pipeline.speech_recognizer.stop_streaming()
            await asyncio.sleep(0.5)  # Brief pause before restarting
            await self.pipeline.speech_recognizer.start_streaming()
            self.google_session_active = True
            logger.info("Reset Google Cloud STT session after error")
            return True
        except Exception as reset_error:
            logger.error(f"Failed to reset Google STT session: {reset_error}")
            self.google_session_active = False
            return False

async def initialize_system():
    """Initialize all system components with Google Cloud Speech integration."""
    global twilio_handler, voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent with Google Cloud Speech...")
    
    # Get Google credentials path
    google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not google_credentials_path:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set in environment")
    else:
        logger.info(f"Using Google credentials from: {google_credentials_path}")
    
    # Initialize Voice AI Agent with Google Cloud Speech
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        credentials_path=google_credentials_path,
        llm_temperature=0.7,
        language='en-US'
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
    
    logger.info("System initialized successfully with knowledge-base agnostic speech enhancements")

@app.route('/', methods=['GET'])
def index():
    """Simple test endpoint."""
    return "Voice AI Agent is running with improved noise handling!"

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls using WebSocket stream."""
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
        # Add call to manager
        twilio_handler.call_manager.add_call(call_sid, from_number, to_number)
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Use WebSocket streaming for real-time conversation
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"Setting up WebSocket stream at: {ws_url}")
        
        # Create the streaming connection with longer media timeout
        connect = Connect()
        connect.add_parameter("timeout", "60") # 60-second media timeout
        stream = Stream(url=ws_url)
        stream.add_parameter("track", "both") # Send audio in both directions
        connect.append(stream)
        response.append(connect)
        
        # Add TwiML to keep connection alive if needed
        response.say("The AI assistant is now listening. Please speak clearly.", voice='alice')
        
        logger.info(f"Generated TwiML for WebSocket streaming: {response}")
        return Response(str(response), mimetype='text/xml')
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        fallback_twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>An error occurred. Please try again later.</Say>
</Response>'''
        return Response(fallback_twiml, mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks with proper cleanup."""
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
        
        # Clean up event loop if call is completed
        if call_status in ['completed', 'failed', 'busy', 'no-answer']:
            # Clean up the event loop for this call
            if call_sid in call_event_loops:
                loop_info = call_event_loops[call_sid]
                # Signal termination
                if 'terminate_flag' in loop_info:
                    loop_info['terminate_flag'].set()
                    
                # Get thread if available
                thread = loop_info.get('thread')
                if thread:
                    # Wait for thread to join with timeout
                    thread.join(timeout=1.0)
                
                try:
                    # Clean up Google STT session if active
                    handler = loop_info.get('handler')
                    if handler and handler.google_session_active:
                        # Create a new event loop for cleanup
                        cleanup_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(cleanup_loop)
                        # Close the STT session
                        cleanup_loop.run_until_complete(handler.pipeline.speech_recognizer.stop_streaming())
                        cleanup_loop.close()
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up STT session: {cleanup_error}")
                
                # Remove from tracking
                call_event_loops.pop(call_sid, None)
                logger.info(f"Cleaned up event loop resources for call {call_sid}")
                
        return Response('', status=204)
    except Exception as e:
        logger.error(f"Error handling status callback: {e}", exc_info=True)
        # Safely remove from dictionary
        call_event_loops.pop(call_sid, None)
        return Response('', status=204)

def run_event_loop_in_thread(loop, ws_handler, ws, call_sid, terminate_flag):
    """Run event loop in a separate thread."""
    try:
        # Set this loop as the event loop for this thread
        asyncio.set_event_loop(loop)
        
        # Create a task for the keep-alive mechanism
        keep_alive_task = asyncio.ensure_future(ws_handler._keep_alive_loop(ws))
        
        # Run the loop until the terminate flag is set or an error occurs
        while not terminate_flag.is_set():
            try:
                # Run the loop for a short duration
                loop.run_until_complete(asyncio.sleep(0.1))
            except Exception as e:
                logger.error(f"Error in event loop for call {call_sid}: {e}")
                break
        
        # Cancel the keep-alive task
        keep_alive_task.cancel()
        try:
            loop.run_until_complete(keep_alive_task)
        except (asyncio.CancelledError, Exception):
            pass
            
        # Close the loop
        loop.close()
        logger.info(f"Event loop for call {call_sid} has been closed")
        
    except Exception as e:
        logger.error(f"Error in event loop thread for call {call_sid}: {e}", exc_info=True)
    finally:
        # Clean up call_event_loops entry if it still exists
        if call_sid in call_event_loops:
            del call_event_loops[call_sid]

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    
    """Handle WebSocket media stream with improved noise handling."""
    logger.info(f"WebSocket connection attempt for call {call_sid}")
    
    if not twilio_handler or not voice_ai_pipeline:
        logger.error("System not initialized")
        return ""
    
    ws = None
    try:
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create WebSocket handler with noise handling optimizations
        ws_handler = WebSocketHandler(call_sid, voice_ai_pipeline)
        
        # Create an event loop for this connection
        loop = asyncio.new_event_loop()
        
        # Create a flag for termination
        terminate_flag = threading.Event()
        
        # Store the event loop and related info
        call_event_loops[call_sid] = {
            'loop': loop,
            'terminate_flag': terminate_flag,
            'handler': ws_handler
        }
        
        # Create and start a thread for the event loop
        loop_thread = threading.Thread(
            target=run_event_loop_in_thread,
            args=(loop, ws_handler, ws, call_sid, terminate_flag),
            daemon=True
        )
        loop_thread.start()
        
        # Add thread to tracking
        call_event_loops[call_sid]['thread'] = loop_thread
        
        # Process connected event in the event loop
        connected_message = json.dumps({
            "event": "connected",
            "protocol": "Call",
            "version": "1.0.0"
        })
        
        # Use asyncio.run_coroutine_threadsafe
        future = asyncio.run_coroutine_threadsafe(
            ws_handler.handle_message(connected_message, ws),
            loop
        )
        # Wait for connected processing to complete
        future.result(timeout=2.0)
        
        # Process messages until connection closed
        while True:
            try:
                # Use longer timeout
                message = ws.receive(timeout=10)
                if message is None:
                    logger.warning(f"Received None message for call {call_sid}")
                    break
                
                # Process the message in the dedicated event loop
                # Wait for processing to complete
                future = asyncio.run_coroutine_threadsafe(
                    ws_handler.handle_message(message, ws),
                    loop
                )
                # Optional: wait with short timeout to ensure message is being processed
                future.result(timeout=0.1)
                
            except ConnectionClosed:
                logger.info(f"WebSocket connection closed for call {call_sid}")
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                # Short pause to avoid CPU spinning on errors
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket cleanup for call {call_sid}")
        
        # Signal termination
        if call_sid in call_event_loops:
            call_event_loops[call_sid]['terminate_flag'].set()
        
        try:
            if ws:
                ws.close()
        except Exception as close_error:
            logger.error(f"Error closing WebSocket: {close_error}")
        
        # Return empty response
        return ""

@app.route('/ws-test')
def ws_test():
    """WebSocket connection test page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Test</title>
        <script>
            function startTest() {
                const wsUrl = document.getElementById('wsUrl').value;
                const output = document.getElementById('output');
                
                output.innerHTML += `<p>Connecting to ${wsUrl}...</p>`;
                
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    output.innerHTML += '<p style="color:green">Connection opened!</p>';
                    ws.send(JSON.stringify({event: 'test', message: 'Hello WebSocket'}));
                };
                
                ws.onmessage = (event) => {
                    output.innerHTML += `<p>Received: ${event.data}</p>`;
                };
                
                ws.onerror = (error) => {
                    output.innerHTML += `<p style="color:red">Error: ${error}</p>`;
                };
                
                ws.onclose = () => {
                    output.innerHTML += '<p>Connection closed</p>';
                };
            }
        </script>
    </head>
    <body>
        <h1>WebSocket Connection Test</h1>
        <input id="wsUrl" type="text" value="wss://your-runpod-url.proxy.runpod.net/ws/test" style="width:400px" />
        <button onclick="startTest()">Test Connection</button>
        <div id="output"></div>
    </body>
    </html>
    """

@app.route('/ws/test', websocket=True)
def ws_test_endpoint():
    """WebSocket test endpoint."""
    try:
        ws = Server.accept(request.environ)
        logger.info("WebSocket test connection established")
        
        ws.send(json.dumps({"status": "connected", "message": "WebSocket test successful!"}))
        
        try:
            while True:
                message = ws.receive(timeout=10)
                if message is None:
                    break
                logger.info(f"Received WebSocket test message: {message}")
                ws.send(json.dumps({"status": "echo", "message": message}))
        except Exception as e:
            logger.error(f"Error in WebSocket test: {e}")
        finally:
            ws.close()
    except Exception as e:
        logger.error(f"Error establishing WebSocket test connection: {e}")
    
    return ""

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