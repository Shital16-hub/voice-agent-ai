#!/usr/bin/env python3
"""
Test script for Google Cloud TTS with different voices and settings.
"""
import os
import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Add the project directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from text_to_speech import GoogleCloudTTS
from text_to_speech.config import config

async def test_voice(voice_name, ssml_rate, test_text):
    """Test a specific voice with given parameters."""
    print(f"\nTesting voice: {voice_name} with rate {ssml_rate}")
    
    # Update config
    config.voice_name = voice_name
    config.ssml_rate = ssml_rate
    
    # Create TTS client
    tts = GoogleCloudTTS(voice_name=voice_name)
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Generate speech with various settings
    audio_data = await tts.synthesize(test_text)
    
    # Save to file
    filename = f"test_output/{voice_name.replace('-', '_')}_rate_{ssml_rate.replace('.', '_')}.wav"
    with open(filename, "wb") as f:
        f.write(audio_data)
    
    print(f"Saved to {filename}")
    return filename

async def main():
    # Test text
    test_text = "Welcome to our telephony system. I'm your AI assistant, ready to help you with any questions you may have about our services and features."
    
    # Voices to test
    voices = [
        "en-US-Neural2-D",  # Male voice, clear and natural
        "en-US-Neural2-F",  # Female voice, professional
        "en-US-Neural2-J",  # Male voice, different style
        "en-US-Polyglot-1"  # Good for telephony
    ]
    
    # Rates to test
    rates = ["1.3", "1.5", "1.8", "2.0"]
    
    print("Starting TTS tests...")
    
    for voice in voices:
        for rate in rates:
            try:
                filename = await test_voice(voice, rate, test_text)
                print(f"✅ Success: {voice} at rate {rate}")
            except Exception as e:
                print(f"❌ Error with {voice} at rate {rate}: {e}")
    
    print("\nTest complete! Check test_output directory for audio files.")
    print("Listen to each file to determine which voice and rate sound best.")

if __name__ == "__main__":
    asyncio.run(main())
