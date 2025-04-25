#!/usr/bin/env python3
"""
Test script for the integrated Voice AI Agent pipeline.
Tests the full STT -> Knowledge Base -> TTS pipeline.
"""
import os
import asyncio
import logging
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the integration components
from integration import TTSIntegration, VoiceAIAgentPipeline

# Import the existing Voice AI Agent components
from voice_ai_agent import VoiceAIAgent
from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR

async def setup_pipeline(args):
    """Set up the pipeline with all components."""
    # Initialize the base Voice AI Agent
    agent = VoiceAIAgent(
        storage_dir=args.storage_dir,
        model_name=args.model_name,
        whisper_model_path=args.whisper_model,
        llm_temperature=args.temperature
    )
    
    # Initialize components
    await agent.init()
    
    # Initialize TTS integration
    tts = TTSIntegration(voice=args.tts_voice)
    await tts.init()
    
    # Create the pipeline
    pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    return pipeline, agent

async def test_pipeline(args):
    """Test the complete STT -> Knowledge Base -> TTS pipeline."""
    print("\n=== Testing Voice AI Agent Pipeline: STT -> KB -> TTS ===\n")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Output path for the generated speech
    if args.output_dir:
        output_speech_file = os.path.join(args.output_dir, "response.mp3")
    else:
        output_speech_file = "response.mp3"
    
    # Set up pipeline
    pipeline, agent = await setup_pipeline(args)
    
    # Get agent stats
    stats = await agent.get_stats()
    print(f"Agent statistics:")
    print(f"- LLM Model: {stats['model_name']}")
    print(f"- Whisper Model: {stats['speech_recognizer_model']}")
    print(f"- Knowledge base documents: {stats['knowledge_base']['document_count']}")
    
    # Run the pipeline
    if args.streaming:
        print(f"\nProcessing audio file with streaming: {args.input_file}")
        
        # Define a callback to handle audio chunks
        chunk_counter = 0
        
        async def audio_callback(audio_chunk):
            nonlocal chunk_counter
            chunk_counter += 1
            
            # Save chunk for verification
            if args.output_dir:
                chunk_file = os.path.join(args.output_dir, f"chunk_{chunk_counter}.mp3")
                with open(chunk_file, "wb") as f:
                    f.write(audio_chunk)
                
                if chunk_counter % 5 == 0:
                    print(f"Saved {chunk_counter} audio chunks...")
        
        # Run streaming pipeline
        result = await pipeline.process_audio_streaming(
            audio_file_path=args.input_file,
            audio_callback=audio_callback
        )
    else:
        # Run standard pipeline
        print(f"\nProcessing audio file: {args.input_file}")
        result = await pipeline.process_audio_file(
            audio_file_path=args.input_file,
            output_speech_file=output_speech_file
        )
    
    # Check for errors
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print results
    print("\n=== Pipeline Results ===")
    print(f"Transcription: {result['transcription']}")
    
    if args.streaming:
        print(f"Response: {result['full_response']}")
        print(f"\nGenerated {result.get('total_chunks', 0)} audio chunks")
        if args.output_dir:
            print(f"Audio chunks saved to: {args.output_dir}")
    else:
        print(f"Response: {result['response']}")
        
        # Print timing information
        print("\n=== Timing Information ===")
        for stage, time_value in result.get('timings', {}).items():
            print(f"{stage.upper()} time: {time_value:.2f} seconds")
        
        # Print output information
        print(f"\nGenerated speech audio saved to: {output_speech_file}")
        print(f"Audio size: {result.get('speech_audio_size', 0)} bytes")
    
    print(f"\nTotal pipeline time: {result.get('total_time', 0):.2f} seconds")
    print("\nPipeline test completed successfully!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the Voice AI Agent integrated pipeline')
    
    # Input and output options
    parser.add_argument('--input-file', required=True, help='Path to input audio file')
    parser.add_argument('--output-dir', default=None, help='Directory to save output files')
    
    # Model options
    parser.add_argument('--storage-dir', default='./storage', help='Storage directory for knowledge base')
    parser.add_argument('--model-name', default='mistral:7b-instruct-v0.2-q4_0', help='LLM model name')
    parser.add_argument('--whisper-model', default='tiny.en', help='Whisper model path')
    parser.add_argument('--tts-voice', default=None, help='TTS voice (Deepgram voice ID)')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature')
    
    # Mode options
    parser.add_argument('--streaming', action='store_true', help='Use streaming mode')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Run the test
    try:
        asyncio.run(test_pipeline(args))
        return 0
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())