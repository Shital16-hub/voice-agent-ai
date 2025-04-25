"""
Speech-to-Text node for the LangGraph-based Voice AI Agent.

This module provides the STT node that processes audio input
and generates transcriptions within the LangGraph flow.
"""
import time
import asyncio
import logging
from typing import Dict, Any, AsyncIterator, Optional, Callable, Awaitable, List

import numpy as np

from integration.stt_integration import STTIntegration
from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR
from speech_to_text.utils.audio_utils import load_audio_file

from langgraph_integration.nodes.state import AgentState, NodeType, ConversationStatus

logger = logging.getLogger(__name__)

class STTNode:
    """
    Speech-to-Text node for LangGraph.
    
    This node processes audio input and generates transcriptions.
    """
    
    def __init__(
        self,
        stt_integration: Optional[STTIntegration] = None,
        speech_recognizer: Optional[StreamingWhisperASR] = None,
        model_path: str = "tiny.en",
        language: str = "en"
    ):
        """
        Initialize the STT node.
        
        Args:
            stt_integration: Existing STT integration to use
            speech_recognizer: Existing speech recognizer to use
            model_path: Path to Whisper model if creating new integration
            language: Language for speech recognition
        """
        if stt_integration:
            self.stt = stt_integration
        elif speech_recognizer:
            self.stt = STTIntegration(speech_recognizer=speech_recognizer, language=language)
        else:
            self.stt = STTIntegration(speech_recognizer=None, language=language)
            # Will need to initialize later
            self.model_path = model_path
            self.initialized = False
    
    async def process(self, state: AgentState) -> AsyncIterator[AgentState]:
        """
        Process the input state and transcribe audio with optimizations.
        
        Args:
            state: The current agent state
            
        Yields:
            Updated agent state with transcription
        """
        # Initialize if needed
        if not getattr(self, 'initialized', True):
            await self.stt.init(model_path=getattr(self, 'model_path', 'tiny.en'))
            self.initialized = True
        
        # If direct text input is provided, bypass STT processing
        if state.text_input:
            state.current_node = NodeType.STT  # Still mark as processed by STT
            state.transcription = state.text_input
            state.query = state.text_input
            state.transcription_confidence = 1.0  # High confidence for direct text
            state.status = ConversationStatus.THINKING
            state.next_node = NodeType.KB
            state.timings["stt"] = 0.0  # Zero processing time
            
            # Add to history
            state.history.append({
                "role": "user",
                "content": state.transcription
            })
            
            # Log bypass
            logger.info(f"STT bypassed with direct text: '{state.text_input}'")
            
            yield state
            return
        
        # Update state
        state.current_node = NodeType.STT
        state.status = ConversationStatus.LISTENING
        
        # Start timing
        start_time = time.time()
        
        try:
            # Process based on input type
            if state.audio_input is not None:
                result = await self._process_audio_data(state)
            elif state.audio_file_path:
                result = await self._process_audio_file(state)
            else:
                # No input provided
                logger.error("No input provided to STT node")
                state.error = "No input provided to STT node"
                state.status = ConversationStatus.ERROR
                yield state
                return
            
            # Update state with result
            state.transcription = result.get("transcription", "")
            if not state.transcription:
                logger.warning("Empty transcription generated or no transcription in result")
                
            # Update additional state
            state.transcription_confidence = result.get("confidence", 1.0)
            if "interim_transcriptions" in result:
                state.interim_transcriptions = result["interim_transcriptions"]
            
            # Set query for KB node
            state.query = state.transcription
            
            # Update status
            state.status = ConversationStatus.THINKING
            state.next_node = NodeType.KB
            
            # Save timing information
            state.timings["stt"] = time.time() - start_time
            
            # Add to history
            state.history.append({
                "role": "user",
                "content": state.transcription
            })
            
            # Debug log the state after processing
            logger.info(f"STT processing complete. Transcription: '{state.transcription}', Confidence: {state.transcription_confidence}")
        
        except Exception as e:
            logger.error(f"Error in STT node: {e}")
            state.error = f"STT error: {str(e)}"
            state.status = ConversationStatus.ERROR
            state.timings["stt"] = time.time() - start_time
        
        # Return updated state
        yield state
    
    async def _process_audio_data(self, state: AgentState) -> Dict[str, Any]:
        """
        Process audio data.
        
        Args:
            state: Current agent state with audio_input
            
        Returns:
            Dictionary with transcription results
        """
        logger.info("Processing audio data")
        
        # Callback for interim results
        async def interim_callback(result):
            if result and result.text:
                logger.debug(f"Interim transcription: {result.text}")
        
        # Process the audio data with timeout protection
        try:
            # Set a timeout for STT processing
            transcribe_task = asyncio.create_task(
                self.stt.transcribe_audio_data(
                    audio_data=state.audio_input,
                    callback=interim_callback
                )
            )
            result = await asyncio.wait_for(transcribe_task, timeout=30.0)  # Increased timeout to 30s
        except asyncio.TimeoutError:
            logger.warning("STT processing timed out")
            # Return a reasonable result with an error flag
            return {
                "transcription": "Audio processing timed out. Please try again with shorter audio.",
                "confidence": 0.0,
                "error": "timeout"
            }
        
        logger.info(f"Generated transcription: {result.get('transcription', '')}")
        return result
    
    async def _process_audio_file(self, state: AgentState) -> Dict[str, Any]:
        """
        Process audio file with increased timeout.
        
        Args:
            state: Current agent state with audio_file_path
            
        Returns:
            Dictionary with transcription results
        """
        logger.info(f"Processing audio file: {state.audio_file_path}")
        
        # Callback for interim results
        async def interim_callback(result):
            if result and result.text:
                logger.debug(f"Interim transcription: {result.text}")
        
        # Process the audio file with timeout protection
        try:
            # Set a timeout for file processing - increase from 15s to 30s for larger files
            transcribe_task = asyncio.create_task(
                self.stt.transcribe_audio_file(
                    audio_file_path=state.audio_file_path,
                    callback=interim_callback
                )
            )
            result = await asyncio.wait_for(transcribe_task, timeout=30.0)  # Increased timeout
        except asyncio.TimeoutError:
            logger.warning(f"Audio file processing timed out: {state.audio_file_path}")
            # Return a reasonable result with an error flag
            return {
                "transcription": "Audio file processing timed out. The file may be too large.",
                "confidence": 0.0,
                "error": "timeout"
            }
        
        logger.info(f"Generated transcription: {result.get('transcription', '')}")
        return result
    
    async def cleanup(self):
        """Clean up resources."""
        logger.debug("Cleaning up STT node")
        # No specific cleanup needed for STT