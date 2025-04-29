#!/usr/bin/env python3
"""
Test script for the aiohttp-based Deepgram streaming client.
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
from speech_to_text.deepgram_stt.streaming import DeepgramStreamingSTT, StreamingTranscriptionResult

async def test_deepgram_aiohttp():
    """Test Deepgram streaming with aiohttp."""
    # Get API key from environment
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("DEEPGRAM_API_KEY not set in environment")
        return False
    
    logger.info("Initializing Deepgram streaming client...")
    client = DeepgramStreamingSTT(
        api_key=api_key,
        model_name="nova-2",
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
        # Test starting the streaming session
        logger.info("Starting streaming session...")
        await client.start_streaming()
        logger.info("Successfully started streaming session!")
        
        # Create some dummy audio data (silence)
        dummy_audio = bytes([0] * 4096)  # 4KB of silence
        
        # Send a chunk of audio
        logger.info("Sending audio chunk...")
        result = await client.process_audio_chunk(dummy_audio, result_callback)
        logger.info(f"Processed audio chunk, result: {result}")
        
        # Stop the streaming session
        logger.info("Stopping streaming session...")
        await client.stop_streaming()
        logger.info("Successfully stopped streaming session!")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Deepgram streaming: {e}")
        try:
            # Try to stop the session in case of error
            await client.stop_streaming()
        except:
            pass
        return False

if __name__ == "__main__":
    success = asyncio.run(test_deepgram_aiohttp())
    if success:
        print("\n✅ Deepgram streaming test with aiohttp PASSED! The connection is working correctly.")
    else:
        print("\n❌ Deepgram streaming test FAILED. See logs for details.")