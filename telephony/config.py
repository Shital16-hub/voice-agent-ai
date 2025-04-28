"""
Configuration settings for telephony integration.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Server Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Audio Configuration
SAMPLE_RATE_TWILIO = 8000  # Twilio's sample rate
SAMPLE_RATE_AI = 16000     # Our AI system's sample rate
CHUNK_SIZE = 320           # 20ms at 8kHz
AUDIO_BUFFER_SIZE = 24000  # 1.5 second buffer - increased for better transcription
MAX_BUFFER_SIZE = 48000    # 3 seconds maximum buffer

# WebSocket Configuration
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_MAX_MESSAGE_SIZE = 1048576  # 1MB

# Performance Settings
SILENCE_THRESHOLD = 0.005   # Reduced threshold to detect more speech
SILENCE_DURATION = 1.0      # seconds - reduced for faster response
MAX_CALL_DURATION = 3600    # 1 hour
MAX_PROCESSING_TIME = 5.0   # Maximum time to spend processing audio (seconds)

# Response Settings
RESPONSE_TIMEOUT = 4.0      # Maximum time to wait for a response (seconds)
MIN_TRANSCRIPTION_LENGTH = 2  # Minimum words in transcription to process

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'