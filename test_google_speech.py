# test_google_speech.py

import os
import asyncio
import logging
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_google_speech():
    logger.info("Testing Google Cloud Speech API")
    
    # Get credentials path
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not set!")
        return False
    
    logger.info(f"Using credentials from: {credentials_path}")
    
    try:
        # Set up credentials
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Create client
        client = speech.SpeechClient(credentials=credentials)
        
        # Create a simple recognition request
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            model="phone_call",
            use_enhanced=True
        )
        
        # Create audio content (simple test audio)
        test_audio = b'\x00' * 3200  # 100ms of silence at 16kHz
        audio = speech.RecognitionAudio(content=test_audio)
        
        # Make request to test API access
        logger.info("Sending test request to Google Cloud Speech API")
        response = client.recognize(config=config, audio=audio)
        
        logger.info("Google Cloud Speech API test successful!")
        logger.info(f"Response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Google Cloud Speech API: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_google_speech())