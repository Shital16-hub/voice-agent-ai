"""
Enhanced configuration settings for telephony integration with improved noise handling.
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
# Increased buffer size for better speech analysis
AUDIO_BUFFER_SIZE = 32000  # 2.0 second buffer (reduced from 38400)
MAX_BUFFER_SIZE = 48000    # 3.0 seconds maximum buffer (reduced from 57600)

# WebSocket Configuration
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_MAX_MESSAGE_SIZE = 1048576  # 1MB

# Enhanced Speech Detection Settings
SILENCE_THRESHOLD = 0.0025   # Increased from 0.0018 for better noise rejection
SILENCE_DURATION = 2.0       # Increased from 1.8 seconds to ensure proper pauses
MAX_CALL_DURATION = 3600     # 1 hour
MAX_PROCESSING_TIME = 5.0    # Reduced from 6.0 seconds

# Response Settings
RESPONSE_TIMEOUT = 4.0        # Reduced from 4.5 seconds
MIN_TRANSCRIPTION_LENGTH = 4  # Increased from 3 to avoid processing noise/short utterances

# Enhanced Noise Filtering Settings
HIGH_PASS_FILTER = 120       # Increased from 100Hz to further reduce low-frequency noise
NOISE_GATE_THRESHOLD = 0.025  # Increased from 0.018
ENABLE_NOISE_FILTERING = True

# Enhanced Barge-in Settings
ENABLE_BARGE_IN = True                   # Enable barge-in functionality
BARGE_IN_THRESHOLD = 0.045               # Increased from 0.045 for more reliable detection
BARGE_IN_DETECTION_WINDOW = 140          # Increased from 100ms for better detection
BARGE_IN_MIN_SPEECH_DURATION = 300       # Increased from 200ms to reduce false positives
BARGE_IN_COOLDOWN_MS = 1500              # 2 second cooldown after agent starts speaking

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# New: Audio Preprocessor Configuration
PREPROCESSOR_ENABLE_DEBUG = False  # Enable detailed debug logging for audio preprocessor
PREPROCESSOR_WARMUP_FRAMES = 15    # Frames to collect before making VAD decisions
PREPROCESSOR_NOISE_FLOOR_MIN = 0.008  # Minimum noise floor level

# STT Optimization Settings - Enhanced for noise handling
STT_INITIAL_PROMPT = """This is a telephone conversation. 
Focus only on the clearly spoken words and ignore any background noise, static, 
beeps, or line interference. Transcribe only the spoken words."""

STT_NO_CONTEXT = True       # Disable context to prevent false additions in noisy environments
STT_TEMPERATURE = 0.0       # Use greedy decoding for less hallucination
STT_PRESET = "default"      # Use default preset with enhanced noise handling