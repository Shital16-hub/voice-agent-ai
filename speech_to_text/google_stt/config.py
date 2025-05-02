"""
Configuration for Google Cloud Speech-to-Text.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Cloud credentials
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# STT settings
LANGUAGE_CODE = os.getenv("STT_LANGUAGE_CODE", "en-US")
SAMPLE_RATE = int(os.getenv("STT_SAMPLE_RATE", "8000"))  # 8kHz for telephony
MODEL = os.getenv("STT_MODEL", "phone_call")
USE_ENHANCED = os.getenv("STT_USE_ENHANCED", "True").lower() == "true"
ENABLE_AUTOMATIC_PUNCTUATION = os.getenv("STT_ENABLE_PUNCTUATION", "True").lower() == "true"

# Speech contexts for better recognition
SPEECH_CONTEXTS = [
    "help", "agent", "cost", "price", "service", 
    "support", "information", "question", "connect"
]
SPEECH_CONTEXT_BOOST = float(os.getenv("STT_CONTEXT_BOOST", "10.0"))

# Barge-in settings
ENABLE_BARGE_IN = os.getenv("STT_ENABLE_BARGE_IN", "True").lower() == "true"
BARGE_IN_THRESHOLD = float(os.getenv("STT_BARGE_IN_THRESHOLD", "0.045"))
BARGE_IN_MIN_DURATION_MS = int(os.getenv("STT_BARGE_IN_MIN_DURATION", "300"))
BARGE_IN_COOLDOWN_MS = int(os.getenv("STT_BARGE_IN_COOLDOWN", "1500"))

# Create a config dictionary
config = {
    "google_credentials_path": GOOGLE_CREDENTIALS_PATH,
    "language_code": LANGUAGE_CODE,
    "sample_rate": SAMPLE_RATE,
    "model": MODEL,
    "use_enhanced": USE_ENHANCED,
    "enable_automatic_punctuation": ENABLE_AUTOMATIC_PUNCTUATION,
    "speech_contexts": SPEECH_CONTEXTS,
    "speech_context_boost": SPEECH_CONTEXT_BOOST,
    "enable_barge_in": ENABLE_BARGE_IN,
    "barge_in_threshold": BARGE_IN_THRESHOLD,
    "barge_in_min_duration_ms": BARGE_IN_MIN_DURATION_MS,
    "barge_in_cooldown_ms": BARGE_IN_COOLDOWN_MS
}