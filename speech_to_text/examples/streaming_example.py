#!/usr/bin/env python3
"""
Example script for streaming speech recognition using Whisper.cpp.
"""
import os
import sys
import time
import argparse
import asyncio
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, PARAMETER_PRESETS
from speech_to_text.utils.audio_utils import load_audio_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_file_in_chunks(
    asr: StreamingWhisperASR,
    audio_file: str,
    chunk_size_ms: int = 1000,
    simulate_realtime: bool = True
) -> tuple[str, float]:
    """
    Process an audio file in chunks to simulate streaming.
    
    Args:
        asr: StreamingWhisperASR instance
        audio_file: Path to audio file
        chunk_size_ms: Size of each chunk in milliseconds
        simulate_realtime: Whether to simulate real-time processing
        
    Returns:
        Tuple of (final_text, processing_time)
    """
    logger.info(f"Processing file: {audio_file}")
    
    try:
        # Load audio file
        audio, sample_rate = load_audio_file(audio_file, target_sr=asr.sample_rate)
        
        # Calculate chunk size in samples
        chunk_size = int(sample_rate * chunk_size_ms / 1000)
        
        # Split audio into chunks
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        
        async def result_callback(result):
            """Callback for handling transcription results."""
            if result.text.strip():
                logger.info(f"[{result.start_time:.2f}s-{result.end_time:.2f}s] {result.text}")
        
        # Process each chunk
        start_time = time.time()
        
        for i in range(num_chunks):
            # Get chunk
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(audio))
            chunk = audio[chunk_start:chunk_end]
            
            # Process chunk
            await asr.process_audio_chunk(chunk, callback=result_callback)
            
            # Simulate real-time processing
            if simulate_realtime and i < num_chunks - 1:
                await asyncio.sleep(chunk_size_ms / 1000)
        
        # Stop streaming and get final text
        final_text, duration = await asr.stop_streaming()
        processing_time = time.time() - start_time
        
        logger.info("\nFinal transcript:")
        logger.info(final_text)
        logger.info(f"\nProcessed {duration:.2f}s of audio in {processing_time:.2f}s "
                  f"({duration/processing_time:.2f}x real-time)")
        
        return final_text, processing_time
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Streaming speech recognition example')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to Whisper model file or model name')
    parser.add_argument('--audio', type=str, required=True,
                      help='Path to audio file')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Chunk size in milliseconds (default: 1000)')
    parser.add_argument('--language', type=str, default='en',
                      help='Language code (default: en)')
    parser.add_argument('--no-realtime', action='store_true',
                      help='Disable real-time simulation')
    parser.add_argument('--threads', type=int, default=4,
                      help='Number of CPU threads to use (default: 4)')
    parser.add_argument('--vad', action='store_true',
                      help='Enable voice activity detection')
    parser.add_argument('--translate', action='store_true',
                      help='Enable translation to English')
    parser.add_argument('--preset', type=str, choices=list(PARAMETER_PRESETS.keys()), 
                      help='Parameter preset to use')
    parser.add_argument('--temperature', type=float,
                      help='Temperature for sampling (higher=more creative)')
    parser.add_argument('--initial-prompt', type=str,
                      help='Initial prompt to guide transcription')
    parser.add_argument('--max-tokens', type=int,
                      help='Maximum tokens per segment (0=no limit)')
    parser.add_argument('--no-context', action='store_true',
                      help='Do not use past transcription as context')
    parser.add_argument('--single-segment', action='store_true',
                      help='Force single segment output (useful for streaming)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Validate inputs
    if not os.path.isfile(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    if args.chunk_size <= 0:
        logger.error("Chunk size must be positive")
        return 1
    
    try:
        # Create parameter dictionary with only provided values
        params = {}
        if args.temperature is not None:
            params['temperature'] = args.temperature
        if args.initial_prompt is not None:
            params['initial_prompt'] = args.initial_prompt
        if args.max_tokens is not None:
            params['max_tokens'] = args.max_tokens
        if args.no_context:
            params['no_context'] = args.no_context
        if args.single_segment:
            params['single_segment'] = args.single_segment
        if args.preset:
            params['preset'] = args.preset
            
        # Create ASR instance
        logger.info(f"Creating StreamingWhisperASR instance with model: {args.model}")
        asr = StreamingWhisperASR(
            model_path=args.model,
            language=args.language,
            n_threads=args.threads,
            chunk_size_ms=2000,
            vad_enabled=args.vad,
            translate=args.translate,
            **params  # Include any parameters that were provided
        )
        
        # Process audio file
        final_text, _ = await process_file_in_chunks(
            asr=asr,
            audio_file=args.audio,
            chunk_size_ms=args.chunk_size,
            simulate_realtime=not args.no_realtime
        )
        
        # Optionally save the transcript
        output_path = f"{args.audio}.txt"
        with open(output_path, 'w') as f:
            f.write(final_text)
        logger.info(f"Transcript saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    try:
        print("Starting streaming speech recognition...")
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)