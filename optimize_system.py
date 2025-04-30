#!/usr/bin/env python3
"""
Optimization script for Voice AI Agent performance.
This script adjusts system parameters for better telephony performance and lower latency.
"""
import os
import sys
import logging
import json
from pathlib import Path
import importlib
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_system():
    """Apply system optimizations for better performance."""
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Check for GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            logger.info(f"Found GPU: {torch.cuda.get_device_name(0)}")
            # Set environment variables for GPU optimization
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        else:
            logger.info("No GPU found, optimizing for CPU usage")
    except ImportError:
        logger.warning("Torch not available, skipping GPU check")
    
    # Set Whisper CPU thread optimization
    os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "4"  # MKL threads
    
    # Check for Deepgram API key
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        logger.warning("DEEPGRAM_API_KEY not found in environment! Voice AI will fall back to Whisper.")
        logger.warning("For best performance, set DEEPGRAM_API_KEY environment variable.")
    else:
        logger.info("Deepgram API key found, using Deepgram for STT")
    
    # Patch config values
    try:
        # Update telephony config
        from telephony.config import (
            SILENCE_THRESHOLD, BARGE_IN_THRESHOLD, 
            MIN_TRANSCRIPTION_LENGTH, AUDIO_BUFFER_SIZE,
            MAX_BUFFER_SIZE
        )
        
        # Log current values
        logger.info(f"Current config: SILENCE_THRESHOLD={SILENCE_THRESHOLD}, "
                   f"BARGE_IN_THRESHOLD={BARGE_IN_THRESHOLD}, "
                   f"MIN_TRANSCRIPTION_LENGTH={MIN_TRANSCRIPTION_LENGTH}, "
                   f"AUDIO_BUFFER_SIZE={AUDIO_BUFFER_SIZE}, "
                   f"MAX_BUFFER_SIZE={MAX_BUFFER_SIZE}")
        
        # Create patch file to update config
        config_path = Path(__file__).parent / "telephony" / "config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Patch the values
            patches = [
                ('SILENCE_THRESHOLD = 0.0018', 'SILENCE_THRESHOLD = 0.0025'),
                ('BARGE_IN_THRESHOLD = 0.045', 'BARGE_IN_THRESHOLD = 0.055'),
                ('MIN_TRANSCRIPTION_LENGTH = 3', 'MIN_TRANSCRIPTION_LENGTH = 4'),
                ('AUDIO_BUFFER_SIZE = 38400', 'AUDIO_BUFFER_SIZE = 32000'),
                ('MAX_BUFFER_SIZE = 57600', 'MAX_BUFFER_SIZE = 48000')
            ]
            
            for old_val, new_val in patches:
                if old_val in config_content:
                    config_content = config_content.replace(old_val, new_val)
            
            # Write back updated config
            with open(config_path, 'w') as f:
                f.write(config_content)
                
            logger.info("Updated telephony config.py with optimized values")
        else:
            logger.warning(f"Config file not found at {config_path}")
    except Exception as e:
        logger.error(f"Error updating config values: {e}")

    # Optimize audio preprocessor parameters
    try:
        # Update audio_preprocessor.py
        audio_preprocessor_path = Path(__file__).parent / "telephony" / "audio_preprocessor.py"
        if audio_preprocessor_path.exists():
            with open(audio_preprocessor_path, 'r') as f:
                ap_content = f.read()
            
            # Patch the values
            ap_patches = [
                ('self.ambient_noise_level = 0.01', 'self.ambient_noise_level = 0.015'),
                ('self.min_noise_floor = 0.005', 'self.min_noise_floor = 0.008'),
                ('self.high_threshold = self.ambient_noise_level * 3.5', 'self.high_threshold = self.ambient_noise_level * 4.5')
            ]
            
            for old_val, new_val in ap_patches:
                if old_val in ap_content:
                    ap_content = ap_content.replace(old_val, new_val)
            
            # Write back updated config
            with open(audio_preprocessor_path, 'w') as f:
                f.write(ap_content)
                
            logger.info("Updated audio_preprocessor.py with optimized values")
        else:
            logger.warning(f"Audio preprocessor file not found at {audio_preprocessor_path}")
    except Exception as e:
        logger.error(f"Error updating audio preprocessor values: {e}")
    
    # Create cache directories if they don't exist
    cache_dirs = [
        "cache",
        "cache/stt_cache",
        "cache/tts_cache"
    ]
    
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Ensured cache directory exists: {cache_dir}")
    
    # Check for model files
    model_dir = Path(__file__).parent / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Check for Whisper model - download base.en if missing
    base_en_path = model_dir / "base.en"
    if not base_en_path.exists():
        logger.info("Whisper base.en model not found, attempting to download...")
        try:
            # Try using huggingface_hub
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id="ggerganov/whisper.cpp", 
                               filename="ggml-base.en.bin", 
                               local_dir=str(model_dir))
                # Rename the file
                (model_dir / "ggml-base.en.bin").rename(base_en_path)
                logger.info("Downloaded Whisper base.en model")
            except ImportError:
                # Fallback to direct download
                import requests
                url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
                logger.info(f"Downloading Whisper model from {url}")
                response = requests.get(url)
                with open(base_en_path, 'wb') as f:
                    f.write(response.content)
                logger.info("Downloaded Whisper base.en model")
        except Exception as e:
            logger.error(f"Failed to download Whisper model: {e}")
    
    # Create .env file with optimized settings if it doesn't exist
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        logger.info("Creating .env file with optimized settings")
        with open(env_path, 'w') as f:
            f.write("""# Optimized settings for Voice AI Agent
LOG_LEVEL=INFO
STT_INITIAL_PROMPT=This is a telephone conversation. Focus only on the clearly spoken words and ignore background noise.
STT_NO_CONTEXT=True
STT_TEMPERATURE=0.0
TTS_ENABLE_CACHING=True
ENABLE_BARGE_IN=True
BARGE_IN_THRESHOLD=0.055
PREPROCESSOR_ENABLE_DEBUG=False
""")
    
    logger.info("Applied system optimizations for improved performance")

if __name__ == "__main__":
    logger.info("Starting Voice AI Agent system optimization")
    optimize_system()
    logger.info("Optimization complete - run your application now for improved performance")