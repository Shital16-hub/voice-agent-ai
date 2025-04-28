"""
Optimized configuration settings for telephony integration with improved noise handling.
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
# Increased buffer size for better noise detection and filtering
AUDIO_BUFFER_SIZE = 32000  # 2 second buffer - increased for better noise analysis
MAX_BUFFER_SIZE = 48000    # 3 seconds maximum buffer

# WebSocket Configuration
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_MAX_MESSAGE_SIZE = 1048576  # 1MB

# Performance Settings - Optimized for noise handling
# Increased silence threshold to better distinguish speech from noise
SILENCE_THRESHOLD = 0.008   # Increased from 0.005 to avoid detecting noise as speech
SILENCE_DURATION = 1.2      # Increased to ensure proper pauses are detected
MAX_CALL_DURATION = 3600    # 1 hour
MAX_PROCESSING_TIME = 5.0   # Maximum time to spend processing audio (seconds)

# Response Settings
RESPONSE_TIMEOUT = 4.0      # Maximum time to wait for a response (seconds)
MIN_TRANSCRIPTION_LENGTH = 3  # Increased from 2 to avoid processing noise/short utterances

# Noise Filtering Settings
HIGH_PASS_FILTER = 80       # High-pass filter cutoff frequency in Hz
NOISE_GATE_THRESHOLD = 0.015  # Noise gate threshold
ENABLE_NOISE_FILTERING = True  # Enable enhanced noise filtering

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# STT Optimization Settings
STT_INITIAL_PROMPT = "This is a clear business conversation. Transcribe the exact words spoken, ignoring background noise."
STT_NO_CONTEXT = True      # Disable context to prevent false additions in noisy environments
STT_TEMPERATURE = 0.0      # Use greedy decoding for less hallucination
STT_PRESET = "default"     # Use default preset with noise handling optimizations