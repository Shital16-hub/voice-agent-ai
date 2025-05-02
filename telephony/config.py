"""
Configuration settings for telephony integration with Google Cloud STT.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Server Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Audio Configuration
SAMPLE_RATE_TWILIO = 8000  # Twilio's sample rate
SAMPLE_RATE_AI = 8000      # Google STT prefers 8kHz for phone_call model
CHUNK_SIZE = 320           # 20ms at 8kHz
AUDIO_BUFFER_SIZE = 4800   # 0.6 second buffer (reduced for faster processing)
MAX_BUFFER_SIZE = 16000    # 2.0 seconds maximum buffer (reduced for lower latency)

# WebSocket Configuration
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_MAX_MESSAGE_SIZE = 1048576  # 1MB

# Speech Detection Settings
SILENCE_THRESHOLD = 0.0025   # Energy threshold for silence detection
SILENCE_DURATION = 1.0       # Reduced from 2.0 seconds for better responsiveness
MAX_CALL_DURATION = 3600     # 1 hour
MAX_PROCESSING_TIME = 4.0    # Reduced from 5.0 seconds for better latency

# Response Settings
RESPONSE_TIMEOUT = 3.0        # Reduced from 4.0 seconds for better latency
MIN_TRANSCRIPTION_LENGTH = 3  # Minimum word count for valid transcriptions

# Barge-in Settings
ENABLE_BARGE_IN = True                   # Enable barge-in functionality
BARGE_IN_THRESHOLD = 0.045               # Energy threshold for barge-in detection
BARGE_IN_DETECTION_WINDOW = 100          # Reduced from 140ms for faster detection
BARGE_IN_MIN_SPEECH_DURATION = 200       # Reduced from 300ms for faster response
BARGE_IN_COOLDOWN_MS = 800               # Reduced from 1500ms for faster response

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Google STT Configuration
STT_LANGUAGE_CODE = "en-US"
STT_MODEL = "phone_call"
STT_USE_ENHANCED = True
STT_ENABLE_AUTOMATIC_PUNCTUATION = True
STT_SPEECH_CONTEXTS = ["help", "price", "plan", "cost", "support", "agent", "stop", "repeat"]
STT_SPEECH_CONTEXT_BOOST = 10.0