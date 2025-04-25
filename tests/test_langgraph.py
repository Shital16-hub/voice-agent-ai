#!/usr/bin/env python3
"""
Test script for the LangGraph-based Voice AI Agent.
"""
import os
import asyncio
import logging
import argparse
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the LangGraph agent
from langgraph_integration import VoiceAILangGraph
from langgraph_integration.config import LangGraphConfig

# Import the base agent for component reuse
from voice_ai_agent import VoiceAIAgent

async def test_langgraph_agent(args):
    """Test the LangGraph-based Voice AI Agent."""
    print("\n=== Testing LangGraph-based Voice AI Agent ===\n")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Output path for the generated speech
    if args.output_dir:
        output_speech_file = os.path.join(args.output_dir, "langgraph_response.mp3")
    else:
        output_speech_file = "langgraph_response.mp3"
    
    # Initialize the base agent for component reuse
    base_agent = VoiceAIAgent(
        storage_dir=args.storage_dir,
        model_name=args.model_name,
        whisper_model_path=args.whisper_model,
        llm_temperature=args.temperature
    )
    
    # Initialize components
    print("Initializing Voice AI Agent components...")
    await base_agent.init()
    
    # Create LangGraph config
    config = LangGraphConfig(
        stt_model=args.whisper_model,
        stt_language="en",
        kb_temperature=args.temperature,
        tts_voice=args.tts_voice,
        debug_mode=True,
        save_state_history=True,
        state_history_path=os.path.join(args.output_dir, "state_history.json") if args.output_dir else None
    )
    
    # Create LangGraph agent
    langgraph_agent = VoiceAILangGraph(
        voice_ai_agent=base_agent,
        config=config
    )
    
    # Audio callback for demonstration
    async def audio_callback(audio_data):
        print(f"Received {len(audio_data)} bytes of audio data")
        
        # Save to a separate file for verification
        if args.output_dir:
            callback_file = os.path.join(args.output_dir, "callback_audio.mp3")
            with open(callback_file, "wb") as f:
                f.write(audio_data)
            print(f"Saved callback audio to {callback_file}")
    
    # Set audio callback
    langgraph_agent.set_audio_callback(audio_callback)
    
    # Initialize the LangGraph agent
    print("Initializing LangGraph...")
    await langgraph_agent.init()
    
    # Process based on input type
    if args.text_input:
        # Process text input
        print(f"\nProcessing text input: {args.text_input}")
        result = await langgraph_agent.process_text(
            text=args.text_input,
            speech_output_path=output_speech_file
        )
    else:
        # Process audio file
        print(f"\nProcessing audio file: {args.input_file}")
        result = await langgraph_agent.process_audio_file(
            audio_file_path=args.input_file,
            speech_output_path=output_speech_file
        )
    
    # Print results
    print("\n=== LangGraph Results ===")
    
    if "error" in result and result["error"]:
        print(f"Error: {result['error']}")
        return
    
    # Safely access transcription
    if "transcription" in result:
        print(f"Transcription: {result['transcription']}")
    else:
        print("Transcription: Not available in results")
        
        # Try to get transcription from the state history if saved
        if args.output_dir and config.save_state_history:
            print("Checking state history for transcription...")
            state_history_path = os.path.join(args.output_dir, "state_history.json")
            if os.path.exists(state_history_path):
                try:
                    with open(state_history_path, 'r') as f:
                        state_history = json.load(f)
                        for state in state_history:
                            if 'transcription' in state and state['transcription']:
                                print(f"Found transcription in state history: {state['transcription']}")
                                break
                except Exception as e:
                    print(f"Error reading state history: {e}")
    
    # Safely access response
    if "response" in result:
        print(f"Response: {result['response']}")
    else:
        print("Response: Not available in results")
    
    # Print timing information
    print("\n=== Timing Information ===")
    for stage, time_value in result.get('timings', {}).items():
        if stage != "start_time":
            print(f"{stage.upper()} time: {time_value:.2f} seconds")
    
    print(f"Total time: {result.get('total_time', 0):.2f} seconds")
    
    # Print output information
    print(f"\nGenerated speech audio saved to: {output_speech_file}")
    if "speech_audio_size" in result:
        print(f"Audio size: {result['speech_audio_size']} bytes")
    
    # Print state tracker summary if available
    if hasattr(langgraph_agent, 'state_tracker'):
        summary = langgraph_agent.state_tracker.get_summary()
        print("\n=== State Tracking Summary ===")
        print(f"Transcriptions: {summary.get('transcriptions', [])}")
        print(f"Responses: {summary.get('responses', [])}")
        print(f"Number of turns: {summary.get('num_turns', 0)}")
    
    # Clean up
    await langgraph_agent.cleanup()
    
    print("\nLangGraph test completed successfully!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the LangGraph-based Voice AI Agent')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-file', help='Path to input audio file')
    input_group.add_argument('--text-input', help='Direct text input')
    
    # Output options
    parser.add_argument('--output-dir', default=None, help='Directory to save output files')
    
    # Model options
    parser.add_argument('--storage-dir', default='./storage', help='Storage directory for knowledge base')
    parser.add_argument('--model-name', default='mistral:7b-instruct-v0.2-q4_0', help='LLM model name')
    parser.add_argument('--whisper-model', default='tiny.en', help='Whisper model path')
    parser.add_argument('--tts-voice', default=None, help='TTS voice (Deepgram voice ID)')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature')
    
    args = parser.parse_args()
    
    # Verify input file exists if specified
    if args.input_file and not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Run the test
    try:
        asyncio.run(test_langgraph_agent(args))
        return 0
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())