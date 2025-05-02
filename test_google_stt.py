#!/usr/bin/env python3
"""
Test script for Google Cloud STT implementation.

This script verifies that the Google Cloud STT integration works correctly with
the updated API for streaming recognition.

Usage:
    python test_google_stt.py --credentials /path/to/credentials.json --audio /path/to/test.wav
"""
import os
import sys
import asyncio
import argparse
import logging
import wave
import io
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_to_text.google_stt import GoogleCloudSTT, STTStreamer
from speech_to_text.utils.audio_utils import load_audio_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_google_stt")

def get_audio_info(audio_path):
    """Get audio file information."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / sample_rate
            
            logger.info(f"Audio file: {audio_path}")
            logger.info(f"  Sample rate: {sample_rate} Hz")
            logger.info(f"  Channels: {channels}")
            logger.info(f"  Sample width: {sample_width} bytes")
            logger.info(f"  Duration: {duration:.2f} seconds")
            
            return {
                "channels": channels,
                "sample_width": sample_width,
                "sample_rate": sample_rate,
                "frames": frames,
                "duration": duration
            }
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        return None

def convert_to_telephony_format(audio_path, output_path=None):
    """
    Convert audio to 8kHz mono format suitable for telephony testing.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to output file (default: input_telephony.wav)
        
    Returns:
        Path to converted file
    """
    try:
        import subprocess
        
        # Default output path if not specified
        if output_path is None:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_telephony.wav"
        
        # Run FFmpeg to convert to 8kHz mono PCM
        cmd = [
            "ffmpeg", "-y",  # Overwrite output file if exists
            "-i", audio_path,
            "-ar", "8000",    # 8kHz sample rate
            "-ac", "1",       # Mono
            "-acodec", "pcm_s16le",  # 16-bit PCM
            output_path
        ]
        
        logger.info("Converting audio to telephony format...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True)
        
        logger.info(f"Converted audio to telephony format: {output_path}")
        
        # Verify conversion
        info = get_audio_info(output_path)
        if info and info["sample_rate"] == 8000 and info["channels"] == 1:
            logger.info("Conversion successful")
            return output_path
        else:
            logger.error("Conversion failed to produce 8kHz mono audio")
            return None
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        logger.info("Continuing with original audio file")
        return audio_path

async def test_transcribe(credentials_path, audio_path):
    """Test basic transcription without streaming."""
    logger.info(f"Testing basic transcription with Google Cloud STT for {audio_path}")

    # Initialize client
    try:
        # Get audio info first
        audio_info = get_audio_info(audio_path)
        sample_rate = audio_info["sample_rate"] if audio_info else None
        
        # Initialize with correct sample rate
        stt_client = GoogleCloudSTT(
            credentials_path=credentials_path,
            language_code="en-US",
            model="phone_call",
            use_enhanced=True,
            sample_rate=sample_rate  # Use detected sample rate
        )
        
        # Load audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        # Transcribe audio
        result = await stt_client.transcribe(audio_data)
        
        # Print result
        logger.info(f"Transcription result: {result.get('transcription', '')}")
        logger.info(f"Confidence: {result.get('confidence', 0)}")
        logger.info(f"Words: {len(result.get('words', []))}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing transcription: {e}")
        return False

async def test_streaming(credentials_path, audio_path):
    """Test streaming transcription."""
    logger.info(f"Testing streaming transcription with Google Cloud STT for {audio_path}")
    
    try:
        # Get audio info first
        audio_info = get_audio_info(audio_path)
        sample_rate = audio_info["sample_rate"] if audio_info else 8000
        
        # Initialize client with the correct sample rate
        stt_client = GoogleCloudSTT(
            credentials_path=credentials_path,
            language_code="en-US",
            model="phone_call",
            use_enhanced=True,
            sample_rate=sample_rate  # Use detected sample rate
        )
        
        # Create streamer
        streamer = STTStreamer(stt_client=stt_client)
        
        # Start streaming
        await streamer.start_streaming()
        
        # Load audio file
        audio, _ = load_audio_file(audio_path, target_sr=sample_rate)
        
        # Convert to bytes (16-bit PCM)
        import numpy as np
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        
        # Process in chunks
        chunk_size = 1600  # 100ms at 16kHz, adjust for other sample rates
        chunk_size = chunk_size * sample_rate // 16000  # Scale for actual sample rate
        results = []
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            result = await streamer.process_audio_chunk(chunk)
            if result:
                logger.info(f"Chunk {i//chunk_size}: {result.get('transcription', '')}")
                results.append(result)
        
        # Stop streaming
        final_result = await streamer.stop_streaming()
        if final_result:
            logger.info(f"Final result: {final_result.get('transcription', '')}")
        
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error testing streaming: {e}")
        return False

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Google Cloud STT implementation")
    parser.add_argument("--credentials", type=str, help="Path to Google Cloud credentials JSON")
    parser.add_argument("--audio", type=str, help="Path to test audio file (WAV format)")
    parser.add_argument("--convert", action="store_true", help="Convert audio to 8kHz telephony format")
    
    args = parser.parse_args()
    
    # Validate arguments
    credentials_path = args.credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        logger.error("No credentials provided. Use --credentials or set GOOGLE_APPLICATION_CREDENTIALS.")
        return 1
    
    audio_path = args.audio
    if not audio_path:
        logger.error("No audio file provided. Use --audio to specify a WAV file.")
        return 1
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return 1
    
    # Get audio information
    audio_info = get_audio_info(audio_path)
    
    # Optionally convert audio to telephony format
    if args.convert:
        converted_path = convert_to_telephony_format(audio_path)
        if converted_path and converted_path != audio_path:
            audio_path = converted_path
            # Get updated audio info
            audio_info = get_audio_info(audio_path)
    
    # Run tests
    logger.info("Starting Google Cloud STT tests...")
    
    # Test basic transcription
    basic_success = await test_transcribe(credentials_path, audio_path)
    logger.info(f"Basic transcription test {'succeeded' if basic_success else 'failed'}")
    
    # Test streaming
    streaming_success = await test_streaming(credentials_path, audio_path)
    logger.info(f"Streaming transcription test {'succeeded' if streaming_success else 'failed'}")
    
    # Overall result
    if basic_success and streaming_success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed.")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)