#!/usr/bin/env python3
"""
Script to check available Deepgram models for your API key.
"""
import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("DEEPGRAM_API_KEY")
if not api_key:
    print("ERROR: DEEPGRAM_API_KEY not found in environment variables")
    exit(1)

async def check_models():
    """Check which Deepgram models are available with your API key."""
    print(f"Checking available models for API key: {api_key[:5]}...{api_key[-5:]}")
    
    # Models to check
    models_to_check = [
        "general",
        "enhanced",
        "nova",
        "nova-2",
        "base",
        "whisper",
        "whisper-medium",
        "whisper-large"
    ]
    
    # Headers for API calls
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    
    # Create a short audio sample
    audio_data = bytes([0] * 4096)  # 4KB of silence
    
    # Test each model
    print("\nTesting models...")
    async with aiohttp.ClientSession() as session:
        available_models = []
        unavailable_models = []
        
        for model in models_to_check:
            try:
                # Make a request to the Deepgram API with this model
                print(f"Testing model: {model}...", end="", flush=True)
                async with session.post(
                    "https://api.deepgram.com/v1/listen",
                    params={"model": model},
                    headers=headers,
                    data=audio_data,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        print(" ✓ AVAILABLE")
                        available_models.append(model)
                    else:
                        error_text = await response.text()
                        print(f" ✗ NOT AVAILABLE - Error: {response.status} - {error_text}")
                        unavailable_models.append((model, response.status, error_text))
            except Exception as e:
                print(f" ✗ ERROR: {str(e)}")
                unavailable_models.append((model, "Error", str(e)))
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY OF AVAILABLE MODELS")
    print("="*50)
    if available_models:
        print("\nAVAILABLE MODELS:")
        for model in available_models:
            print(f"  - {model}")
    else:
        print("\nNo models available with this API key.")
    
    print("\nRECOMMENDED MODEL CONFIGURATION:")
    if "nova-2" in available_models:
        print("  Use 'nova-2' (best quality)")
    elif "nova" in available_models:
        print("  Use 'nova'")
    elif "enhanced" in available_models:
        print("  Use 'enhanced'")
    elif "whisper-large" in available_models:
        print("  Use 'whisper-large'")
    elif "whisper-medium" in available_models:
        print("  Use 'whisper-medium'")
    elif "whisper" in available_models:
        print("  Use 'whisper'")
    elif available_models:
        print(f"  Use '{available_models[0]}'")
    else:
        print("  No models available - please check your API key")
    
    print("\nTo update your configuration:")
    print("1. Edit the config.py file:")
    print("   vi /workspace/voice-ai/voice-agent-ai/speech_to_text/config.py")
    print("\n2. Find and update the model_name field with your preferred model")

# Run the async function
if __name__ == "__main__":
    asyncio.run(check_models())