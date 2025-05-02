"""
End-to-end pipeline orchestration for Voice AI Agent with Google Cloud STT.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

import numpy as np

from speech_to_text.google_stt import GoogleCloudSTT
from integration.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.query_engine import QueryEngine
from integration.tts_integration import TTSIntegration

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline orchestration for Voice AI Agent.
    
    Provides a high-level interface for running the complete
    STT -> Knowledge Base -> TTS pipeline with Google Cloud STT.
    """
    
    def __init__(
        self,
        speech_recognizer: Union[GoogleCloudSTT, Any],
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """
        Initialize the pipeline with existing components.
        
        Args:
            speech_recognizer: Initialized STT component (Google Cloud STT or other)
            conversation_manager: Initialized conversation manager
            query_engine: Initialized query engine
            tts_integration: Initialized TTS integration
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create a helper for filtering out non-speech transcriptions
        self.stt_integration = STTIntegration(speech_recognizer)
        
        # Determine if we're using Google Cloud STT
        self.using_google_stt = isinstance(speech_recognizer, GoogleCloudSTT)
        logger.info(f"Pipeline initialized with {'Google Cloud' if self.using_google_stt else 'Other'} STT")
    
    async def _is_valid_transcription(self, transcription: str) -> bool:
        """
        Check if a transcription is valid and should be processed.
        
        Args:
            transcription: The transcription text
            
        Returns:
            True if the transcription is valid
        """
        # First clean up the transcription
        cleaned_text = self.stt_integration.cleanup_transcription(transcription)
        
        # If it's empty after cleaning, it's not valid
        if not cleaned_text:
            return False
            
        # Use STT integration's validation
        return self.stt_integration.is_valid_transcription(cleaned_text)
    

    
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
        
        try:
            # Transcribe using STT integration
            result = await self.stt_integration.transcribe_audio_file(audio_file_path)
            transcription = result.get("transcription", "")
            
            # Validate transcription
            is_valid = await self._is_valid_transcription(transcription)
            if not is_valid:
                logger.warning(f"Transcription not valid for processing: '{transcription}'")
                return {"error": "No valid transcription detected", "transcription": transcription}
                
            timings["stt"] = time.time() - stt_start
            logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
            
        except Exception as e:
            logger.error(f"Error in STT stage: {e}")
            return {"error": f"STT error: {str(e)}"}
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        try:
            # Retrieve context and generate response
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
        audio_data: Union[bytes, np.ndarray],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process audio data with streaming response directly to speech.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            audio_callback: Callback to handle audio data
            
        Returns:
            Dictionary with stats about the process
        """
        logger.info(f"Starting streaming pipeline with audio: {type(audio_data)}")
        
        # Record start time for tracking
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        try:
            # Ensure audio is in the right format for Google Cloud STT
            if isinstance(audio_data, np.ndarray):
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = audio_data
            
            # Transcribe audio
            result = await self.stt_integration.transcribe_audio_data(audio_bytes)
            transcription = result.get("transcription", "")
            
            # Validate transcription
            is_valid = await self._is_valid_transcription(transcription)
            if not is_valid:
                logger.warning(f"Transcription not valid for processing: '{transcription}'")
                return {"error": "No valid transcription detected", "transcription": transcription}
                
            logger.info(f"Transcription: {transcription}")
            transcription_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"error": f"Transcription error: {str(e)}"}
        
        # Stream the response with TTS
        try:
            # Stream the response directly to TTS
            total_chunks = 0
            total_audio_bytes = 0
            response_start_time = time.time()
            full_response = ""
            
            # Use the query engine's streaming method
            async for chunk in self.query_engine.query_with_streaming(transcription):
                chunk_text = chunk.get("chunk", "")
                
                if chunk_text:
                    # Add to full response
                    full_response += chunk_text
                    
                    # Convert to speech and send to callback
                    audio_data = await self.tts_integration.text_to_speech(chunk_text)
                    await audio_callback(audio_data)
                    
                    # Update stats
                    total_chunks += 1
                    total_audio_bytes += len(audio_data)
            
            # Calculate stats
            response_time = time.time() - response_start_time
            total_time = time.time() - start_time
            
            return {
                "transcription": transcription,
                "transcription_time": transcription_time,
                "response_time": response_time,
                "total_time": total_time,
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "full_response": full_response
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            return {
                "error": f"Streaming error: {str(e)}",
                "transcription": transcription,
                "transcription_time": transcription_time
            }
    
    async def process_realtime_stream(
        self,
        audio_chunk_generator: AsyncIterator[np.ndarray],
        audio_output_callback: Callable[[bytes], Awaitable[None]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a real-time audio stream with immediate response.
        
        This method is designed for WebSocket-based streaming where audio chunks
        are continuously arriving and responses should be sent back as soon as possible.
        
        Args:
            audio_chunk_generator: Async generator producing audio chunks
            audio_output_callback: Callback to handle output audio data
            
        Yields:
            Status updates and results
        """
        logger.info("Starting real-time audio stream processing")
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # Start streaming session
        await self.stt_integration.start_streaming()
        
        # Timing stats
        start_time = time.time()
        
        try:
            # Process incoming audio chunks
            async for audio_chunk in audio_chunk_generator:
                # Convert to bytes for STT if needed
                if isinstance(audio_chunk, np.ndarray):
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = audio_chunk
                
                # Process through STT with streaming
                result = await self.stt_integration.process_stream_chunk(audio_bytes)
                
                # Process transcription result if available
                if result and result.get("is_final", False):
                    transcription = result.get("transcription", "")
                    
                    # Check if valid transcription
                    if result.get("is_valid", False) and transcription:
                        # Yield status update
                        yield {
                            "status": "transcribed",
                            "transcription": transcription
                        }
                        
                        # Query knowledge base
                        query_result = await self.query_engine.query(transcription)
                        response = query_result.get("response", "")
                        
                        if response:
                            # Convert to speech
                            speech_audio = await self.tts_integration.text_to_speech(response)
                            
                            # Send through callback
                            await audio_output_callback(speech_audio)
                            
                            # Yield response
                            yield {
                                "status": "response",
                                "transcription": transcription,
                                "response": response,
                                "audio_size": len(speech_audio)
                            }
                
                # Check for barge-in
                if result and result.get("barge_in_detected", False):
                    yield {
                        "status": "barge_in",
                        "transcription": result.get("transcription", "")
                    }
            
            # End streaming session
            final_transcription, _ = await self.stt_integration.end_streaming()
            
            # Process final transcription if valid
            if final_transcription and await self._is_valid_transcription(final_transcription):
                # Generate final response
                query_result = await self.query_engine.query(final_transcription)
                final_response = query_result.get("response", "")
                
                if final_response:
                    # Convert to speech
                    final_speech = await self.tts_integration.text_to_speech(final_response)
                    
                    # Send through callback
                    await audio_output_callback(final_speech)
                    
                    # Yield final response
                    yield {
                        "status": "final",
                        "transcription": final_transcription,
                        "response": final_response,
                        "audio_size": len(final_speech),
                        "total_time": time.time() - start_time
                    }
            
            # Yield completion
            yield {
                "status": "complete",
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in real-time stream processing: {e}")
            yield {
                "status": "error",
                "error": str(e),
                "total_time": time.time() - start_time
            }
        finally:
            # Ensure STT streaming session is closed
            await self.stt_integration.end_streaming()