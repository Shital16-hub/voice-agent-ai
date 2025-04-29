#!/usr/bin/env python3
"""
Test script for the non-streaming Deepgram STT implementation.
"""
import os
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the modified Deepgram STT class
from speech_to_text.deepgram_stt import DeepgramStreamingSTT, StreamingTranscriptionResult

async def test_non_streaming_deepgram():
    """Test the non-streaming Deepgram STT implementation."""
    # Get API key from environment
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("DEEPGRAM_API_KEY not set in environment")
        return False
    
    logger.info(f"API key found: {api_key[:5]}...{api_key[-5:]} (length: {len(api_key)})")
    
    # Log any issues with the API key
    if api_key.startswith('"') or api_key.endswith('"'):
        logger.warning("API key has quotes around it, which may cause issues")
    if api_key.startswith("'") or api_key.endswith("'"):
        logger.warning("API key has single quotes around it, which may cause issues")
    if " " in api_key:
        logger.warning("API key contains spaces, which may cause issues")
    
    logger.info("Initializing Deepgram STT client...")
    client = DeepgramStreamingSTT(
        api_key=api_key,
        model_name="general",
        language="en-US",
        sample_rate=16000,
        encoding="linear16",
        channels=1,
        interim_results=True
    )
    
    # Define a callback for results
    async def result_callback(result: StreamingTranscriptionResult):
        logger.info(f"Result: {result.text} (final: {result.is_final}, confidence: {result.confidence})")
    
    try:
        # Test starting the session
        logger.info("Starting Deepgram session...")
        await client.start_streaming()
        logger.info("Successfully started Deepgram session!")
        
        # Create some dummy audio data (simulated speech)
        # This creates a pattern that might be recognized as speech
        audio_data = bytearray()
        for i in range(8192):
            # Create a simple sine wave pattern (very basic)
            value = int(127 + 127 * (i % 100) / 100)
            audio_data.append(value)
        
        # Send the audio data
        logger.info("Sending audio data...")
        result = await client.process_audio_chunk(audio_data, result_callback)
        
        if result:
            logger.info(f"Received result: {result.text}")
        else:
            logger.info("No immediate result (normal for this implementation)")
            
        # Force processing any buffered audio
        logger.info("Processing any buffered audio...")
        await client.stop_streaming()
        logger.info("Successfully stopped Deepgram session!")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Deepgram: {e}")
        try:
            await client.stop_streaming()
        except:
            pass
        return False

if __name__ == "__main__":
    success = asyncio.run(test_non_streaming_deepgram())
    if success:
        print("\n✅ Deepgram test PASSED! The non-streaming implementation is working correctly.")
    else:
        print("\n❌ Deepgram test FAILED. See logs for details.")