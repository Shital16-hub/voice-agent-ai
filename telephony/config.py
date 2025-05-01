"""
Enhanced configuration settings for telephony integration with improved barge-in.
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
# Reduced buffer sizes for faster processing
AUDIO_BUFFER_SIZE = 24000  # 1.5 second buffer (reduced from 32000)
MAX_BUFFER_SIZE = 32000    # 2.0 seconds maximum buffer (reduced from 48000)

# WebSocket Configuration
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_MAX_MESSAGE_SIZE = 1048576  # 1MB

# Enhanced Speech Detection Settings - More Sensitive
SILENCE_THRESHOLD = 0.0018   # Reduced from 0.0025 for better sensitivity
SILENCE_DURATION = 1.5       # Reduced from 2.0 seconds for faster response
MAX_CALL_DURATION = 3600     # 1 hour
MAX_PROCESSING_TIME = 4.0    # Reduced from 5.0 seconds

# Response Settings
RESPONSE_TIMEOUT = 3.0        # Reduced from 4.0 seconds
MIN_TRANSCRIPTION_LENGTH = 3  # Reduced from 4 to process shorter utterances

# Enhanced Noise Filtering Settings
HIGH_PASS_FILTER = 120
NOISE_GATE_THRESHOLD = 0.02  # Reduced from 0.025 for better sensitivity
ENABLE_NOISE_FILTERING = True

# Enhanced Barge-in Settings
ENABLE_BARGE_IN = True
BARGE_IN_THRESHOLD = 0.04               # Reduced from 0.045 for faster detection
BARGE_IN_DETECTION_WINDOW = 100         # Reduced from 140ms for faster detection
BARGE_IN_MIN_SPEECH_DURATION = 200      # Reduced from 300ms for faster response
BARGE_IN_COOLDOWN_MS = 500              # Reduced from 1500ms to 500ms

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# New: Audio Preprocessor Configuration - Optimized for barge-in
PREPROCESSOR_ENABLE_DEBUG = False  # Enable detailed debug logging for audio preprocessor
PREPROCESSOR_WARMUP_FRAMES = 8     # Reduced from 15 for faster startup
PREPROCESSOR_NOISE_FLOOR_MIN = 0.006  # Reduced from 0.008 for better sensitivity

# STT Optimization Settings
STT_INITIAL_PROMPT = """This is a telephone conversation. 
Focus only on the clearly spoken words and ignore any background noise or interference."""

STT_NO_CONTEXT = True
STT_TEMPERATURE = 0.0
STT_PRESET = "default"