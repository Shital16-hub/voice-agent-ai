"""
Text-to-Speech node for the LangGraph-based Voice AI Agent.

This module provides the TTS node that processes text responses
and generates speech output within the LangGraph flow.
"""
import os
import time
import logging
import asyncio
from typing import Dict, Any, AsyncIterator, Optional, Callable, Awaitable

from integration.tts_integration import TTSIntegration
from text_to_speech import DeepgramTTS

from langgraph_integration.nodes.state import AgentState, NodeType, ConversationStatus

logger = logging.getLogger(__name__)

class TTSNode:
    """
    Text-to-Speech node for LangGraph.
    
    This node processes text responses and generates speech output.
    """
    
    def __init__(
        self,
        tts_integration: Optional[TTSIntegration] = None,
        tts_client: Optional[DeepgramTTS] = None,
        voice: Optional[str] = None,
        output_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
    ):
        """
        Initialize the TTS node.
        
        Args:
            tts_integration: Existing TTS integration to use
            tts_client: TTS client to use if creating new integration
            voice: Voice to use if creating new integration
            output_callback: Callback for audio output (e.g., for telephony)
        """
        if tts_integration:
            self.tts = tts_integration
        elif tts_client:
            self.tts = TTSIntegration(tts_client=tts_client)
        else:
            self.tts = TTSIntegration(voice=voice)
            # Will need to initialize later
            self.initialized = False
        
        self.output_callback = output_callback
    
    async def process(self, state: AgentState) -> AsyncIterator[AgentState]:
        """
        Process the input state and generate speech with timeout protection.
        
        Args:
            state: The current agent state
            
        Yields:
            Updated agent state with speech output
        """
        # Initialize if needed
        if not getattr(self, 'initialized', True):
            await self.tts.init()
            self.initialized = True
        
        # Update state
        state.current_node = NodeType.TTS
        state.status = ConversationStatus.RESPONDING
        
        # Start timing
        start_time = time.time()
        
        try:
            # Check for response
            if not state.response:
                logger.error("No response provided to TTS node")
                state.error = "No response provided to TTS node"
                state.status = ConversationStatus.ERROR
                yield state
                return
            
            # Log the response being converted to speech
            logger.info(f"Converting response to speech: '{state.response[:100]}...'")
            
            # Convert response to speech with timeout protection
            try:
                # Use asyncio.wait_for to enforce a timeout
                tts_task = asyncio.create_task(self.tts.text_to_speech(state.response))
                speech_data = await asyncio.wait_for(tts_task, timeout=10.0)  # 10-second timeout
            except asyncio.TimeoutError:
                logger.warning("TTS processing timed out")
                # Create some minimal speech data as a fallback
                speech_data = await self._generate_fallback_speech(
                    "I'm sorry, but speech synthesis is taking longer than expected. Please try again later."
                )
            
            # Save speech data in state
            state.speech_output = speech_data
            
            # Save to file if path provided
            if state.speech_output_path:
                # Make sure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(state.speech_output_path)), exist_ok=True)
                
                # Save file
                with open(state.speech_output_path, "wb") as f:
                    f.write(speech_data)
                    
                logger.info(f"Saved speech to {state.speech_output_path}")
            
            # Call output callback if provided (e.g., for telephony)
            if self.output_callback:
                await self.output_callback(speech_data)
            
            # Update status
            state.status = ConversationStatus.COMPLETED
            state.next_node = None  # End of flow
            
            # Save timing information
            state.timings["tts"] = time.time() - start_time
            
            # Debug log the state after processing
            logger.info(f"TTS processing complete. Generated {len(speech_data)} bytes of audio")
            
        except Exception as e:
            logger.error(f"Error in TTS node: {e}")
            state.error = f"TTS error: {str(e)}"
            state.status = ConversationStatus.ERROR
            
            # Still need to record timing
            state.timings["tts"] = time.time() - start_time
        
        # Return updated state
        yield state
    
    async def _generate_fallback_speech(self, message: str) -> bytes:
        """
        Generate fallback speech in case of errors or timeouts.
        
        Args:
            message: Message to convert to speech
            
        Returns:
            Audio data as bytes
        """
        try:
            # Try to generate speech with a short timeout
            tts_task = asyncio.create_task(self.tts.text_to_speech(message))
            return await asyncio.wait_for(tts_task, timeout=3.0)
        except Exception as e:
            logger.error(f"Error generating fallback speech: {e}")
            # Return minimal MP3 data as a last resort
            # This is just a placeholder - in production you might want to have pre-generated audio files
            return b"\x00" * 1000
    
    async def cleanup(self):
        """Clean up resources."""
        logger.debug("Cleaning up TTS node")
        if hasattr(self, 'tts') and self.tts:
            await self.tts.cleanup()