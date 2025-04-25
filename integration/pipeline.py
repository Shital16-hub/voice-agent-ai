"""
End-to-end pipeline orchestration for Voice AI Agent.

This module provides high-level functions for running the complete
STT -> Knowledge Base -> TTS pipeline.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

import numpy as np

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.query_engine import QueryEngine

from integration.tts_integration import TTSIntegration

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline orchestration for Voice AI Agent.
    
    Provides a high-level interface for running the complete
    STT -> Knowledge Base -> TTS pipeline.
    """
    
    def __init__(
        self,
        speech_recognizer: StreamingWhisperASR,
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """
        Initialize the pipeline with existing components.
        
        Args:
            speech_recognizer: Initialized STT component
            conversation_manager: Initialized conversation manager
            query_engine: Initialized query engine
            tts_integration: Initialized TTS integration
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        output_speech_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            audio_file_path: Path to the input audio file
            output_speech_file: Path to save the output speech file (optional)
            
        Returns:
            Dictionary with results from each stage
        """
        logger.info(f"Starting end-to-end pipeline with audio: {audio_file_path}")
        
        # Track timing for each stage
        timings = {}
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {"error": f"Error loading audio file: {e}"}
        
        # Process for transcription
        transcription, duration = await self._transcribe_audio(audio)
        
        if not transcription.strip():
            return {"error": "No transcription detected"}
            
        timings["stt"] = time.time() - stt_start
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        try:
            # Retrieve context and generate response
            retrieval_results = await self.query_engine.retrieve_with_sources(transcription)
            query_result = await self.query_engine.query(transcription)
            response = query_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech
        logger.info("STAGE 3: Text-to-Speech")
        tts_start = time.time()
        
        try:
            # Convert response to speech
            speech_audio = await self.tts_integration.text_to_speech(response)
            
            # Save speech audio if output file specified
            if output_speech_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_speech_file)), exist_ok=True)
                with open(output_speech_file, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {output_speech_file}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
        except Exception as e:
            logger.error(f"Error in TTS stage: {e}")
            return {
                "error": f"TTS error: {str(e)}",
                "transcription": transcription,
                "response": response
            }
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"End-to-end pipeline completed in {total_time:.2f}s")
        
        # Compile results
        return {
            "transcription": transcription,
            "response": response,
            "speech_audio_size": len(speech_audio),
            "speech_audio": None if output_speech_file else speech_audio,
            "timings": timings,
            "total_time": total_time
        }
    
    async def process_audio_streaming(
        self,
        audio_data: Union[np.ndarray, bytes],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process audio data through the pipeline with streaming output.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            audio_callback: Callback to handle output audio
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Convert audio data to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio_array = audio_data
            
            # Start speech recognition
            logger.info("Starting speech recognition...")
            self.speech_recognizer.start_streaming()
            
            # Process audio chunk
            result = await self.speech_recognizer.process_audio_chunk(audio_array)
            
            # Get final transcription
            transcription = ""
            if result and result.text:
                transcription = result.text
            else:
                # Try to get from stop_streaming
                final_text, duration = await self.speech_recognizer.stop_streaming()
                transcription = final_text
            
            if not transcription or transcription.strip() == "":
                logger.warning("No transcription detected")
                return {
                    "error": "No speech detected",
                    "transcription": "",
                    "response": ""
                }
            
            logger.info(f"Transcription: {transcription}")
            
            # Process through conversation manager and query engine
            logger.info("Querying knowledge base...")
            
            # Use the conversation manager to process the query
            response_data = await self.conversation_manager.handle_user_input(transcription)
            response = response_data.get("response", "")
            
            if not response:
                logger.warning("No response from knowledge base")
                return {
                    "error": "No response generated",
                    "transcription": transcription,
                    "response": ""
                }
            
            logger.info(f"Response: {response}")
            
            # Convert to speech and send via callback
            logger.info("Converting response to speech...")
            speech_audio = await self.tts_integration.text_to_speech(response)
            
            if speech_audio:
                await audio_callback(speech_audio)
                logger.info("Speech audio sent")
            
            return {
                "transcription": transcription,
                "response": response,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}", exc_info=True)
            return {
                "error": str(e),
                "total_time": time.time() - start_time
            }
        finally:
            # Make sure to stop streaming
            try:
                if self.speech_recognizer.is_streaming:
                    await self.speech_recognizer.stop_streaming()
            except:
                pass
    
    async def _transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio data.
        
        This method handles both short and longer audio files,
        with appropriate parameter adjustments.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        # Try multiple approaches to get transcription
        transcription = ""
        duration = 0
        
        # Save original VAD setting
        original_vad = self.speech_recognizer.vad_enabled
        
        # Disable VAD for short audio
        self.speech_recognizer.vad_enabled = False
        
        # First attempt
        try:
            self.speech_recognizer.start_streaming()
            await self.speech_recognizer.process_audio_chunk(audio)
            transcription, duration = await self.speech_recognizer.stop_streaming()
        except Exception as e:
            logger.warning(f"First transcription attempt failed: {e}")
        
        # If first attempt failed, try again
        if not transcription or transcription.strip() == "":
            try:
                logger.info("First attempt yielded no transcription, trying again")
                self.speech_recognizer.start_streaming()
                await self.speech_recognizer.process_audio_chunk(audio)
                transcription, duration = await self.speech_recognizer.stop_streaming()
            except Exception as e:
                logger.warning(f"Second transcription attempt failed: {e}")
        
        # Restore original VAD setting
        self.speech_recognizer.vad_enabled = original_vad
        
        return transcription, duration