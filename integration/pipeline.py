"""
End-to-end pipeline orchestration for Voice AI Agent with Google Cloud Speech.

This module provides high-level functions for running the complete
STT -> Knowledge Base -> TTS pipeline with Google Cloud STT integration.
"""
import os
import asyncio
import logging
import time


import numpy as np
from scipy import signal

# Updated imports for Google Cloud Speech

from speech_to_text.google_cloud_stt import StreamingTranscriptionResult

from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.query_engine import QueryEngine

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from speech_to_text.stt_integration import STTIntegration
from typing import Dict, Any, Callable, Awaitable, Optional, List, Union, AsyncIterator

from integration.tts_integration import TTSIntegration

# Minimum word count for a valid user query
MIN_VALID_WORDS = 2

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline orchestration for Voice AI Agent.
    
    Provides a high-level interface for running the complete
    STT -> Knowledge Base -> TTS pipeline with Google Cloud Speech.
    """
    
    def __init__(
        self,
        speech_recognizer: Union[GoogleCloudStreamingSTT, Any],
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """
        Initialize the pipeline with existing components.
        
        Args:
            speech_recognizer: Initialized STT component (Google Cloud or other)
            conversation_manager: Initialized conversation manager
            query_engine: Initialized query engine
            tts_integration: Initialized TTS integration
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create a helper for filtering out non-speech transcriptions
        self.stt_helper = STTIntegration(speech_recognizer)
        
        # Determine if we're using Google Cloud STT
        self.using_google = isinstance(speech_recognizer, GoogleCloudStreamingSTT)
        logger.info(f"Pipeline initialized with {'Google Cloud Speech' if self.using_google else 'Other'} STT")
    
    async def _is_valid_transcription(self, transcription: str) -> bool:
        """
        Check if a transcription is valid and should be processed.
        
        Args:
            transcription: The transcription text
            
        Returns:
            True if the transcription is valid
        """
        # First clean up the transcription
        cleaned_text = self.stt_helper.cleanup_transcription(transcription)
        
        # If it's empty after cleaning, it's not valid
        if not cleaned_text:
            return False
            
        # Check if it has enough words
        words = cleaned_text.split()
        if len(words) < MIN_VALID_WORDS:
            return False
            
        return True
    
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
        
        # Log audio file info
        import os
        logger.info(f"Audio file size: {os.path.getsize(audio_file_path)} bytes")
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
            logger.info(f"Loaded audio: {len(audio)} samples, {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Error loading audio file: {e}", exc_info=True)
            return {"error": f"Error loading audio file: {e}"}
        
        # Process for transcription
        logger.info("Transcribing audio...")
        transcription, duration = await self._transcribe_audio(audio)
        
        # Validate transcription
        is_valid = await self._is_valid_transcription(transcription)
        if not is_valid:
            logger.warning(f"Transcription not valid for processing: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
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
            # Ensure audio is in the right format
            if isinstance(audio_data, bytes):
                # Convert bytes to numpy array if needed
                audio = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Transcribe audio
            transcription, duration = await self._transcribe_audio(audio)
            
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
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        speech_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio data through the complete pipeline.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            speech_output_path: Path to save speech output
            
        Returns:
            Results dictionary
        """
        logger.info(f"Starting pipeline with audio data: {type(audio_data)}")
        
        # Track timing
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # Convert audio data to numpy array if needed
        if isinstance(audio_data, bytes):
            audio = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio = audio_data
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Process for transcription
        transcription, duration = await self._transcribe_audio(audio)
        
        # Validate transcription
        is_valid = await self._is_valid_transcription(transcription)
        if not is_valid:
            logger.warning(f"Transcription not valid for processing: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings = {"stt": time.time() - stt_start}
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
            if speech_output_path:
                os.makedirs(os.path.dirname(os.path.abspath(speech_output_path)), exist_ok=True)
                with open(speech_output_path, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {speech_output_path}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Compile results
            return {
                "transcription": transcription,
                "response": response,
                "speech_audio_size": len(speech_audio),
                "speech_audio": speech_audio,
                "timings": timings,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in TTS stage: {e}")
            return {
                "error": f"TTS error: {str(e)}",
                "transcription": transcription,
                "response": response
            }
    
    async def _transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio data using Google Cloud Speech or other STT engine.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        logger.info(f"Transcribing audio: {len(audio)} samples")
        
        # Check if we're using Google Cloud STT
        if self.using_google:
            return await self._transcribe_audio_google(audio)
        else:
            # Use a more generic approach for other STT engines
            return await self._transcribe_audio_generic(audio)
    
    async def _transcribe_audio_google(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio using Google Cloud Speech API.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        try:
            # Normalize audio data to ensure proper range
            audio = np.clip(audio, -1.0, 1.0)
            
            # Apply preprocessing for better speech recognition
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(4, 100/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # Apply band-pass filter for telephony frequency range
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            filtered_audio = signal.filtfilt(b, a, filtered_audio)
            
            # Convert to 16-bit PCM bytes
            audio_bytes = (filtered_audio * 32767).astype(np.int16).tobytes()
            
            # Start a streaming session
            await self.speech_recognizer.start_streaming()
            
            # Track final results
            final_results = []
            
            # Process callback to collect results
            async def collect_result(result):
                if result.is_final:
                    final_results.append(result)
            
            # Process audio in chunks for better management
            chunk_size = 4096  # ~128ms at 16kHz
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                result = await self.speech_recognizer.process_audio_chunk(chunk, collect_result)
                
                # Add final results directly
                if result and result.is_final:
                    final_results.append(result)
            
            # Stop streaming
            transcription, duration = await self.speech_recognizer.stop_streaming()
            
            # If we didn't get a result from stop_streaming but have final results
            if not transcription and final_results:
                # Use the result with highest confidence
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                # Calculate duration if possible
                duration = best_result.end_time - best_result.start_time if best_result.end_time > 0 else len(audio) / 16000
            elif not transcription:
                # No results at all
                transcription = ""
                duration = len(audio) / 16000
            
            # Clean up the transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in Google Cloud transcription: {e}", exc_info=True)
            return "", len(audio) / 16000
    
    async def _transcribe_audio_generic(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Generic transcription method for other STT engines.
        """
        try:
            # Reset any existing streaming session
            if hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming:
                await self.speech_recognizer.stop_streaming()
            
            # Start a new streaming session
            if hasattr(self.speech_recognizer, 'start_streaming'):
                self.speech_recognizer.start_streaming()
            
            # Process audio data
            await self.speech_recognizer.process_audio_chunk(audio)
            
            # Get final transcription
            transcription, duration = await self.speech_recognizer.stop_streaming()
            
            # Clean up transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            # Check if we got a valid transcription
            if not transcription or transcription.strip() == "" or transcription == "[BLANK_AUDIO]":
                logger.warning("No valid transcription obtained")
                transcription = ""
                duration = len(audio) / 16000  # Estimate duration
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)
            transcription = ""
            duration = len(audio) / 16000  # Estimate duration
        
        return transcription, duration
    
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
        
        # Track state
        accumulated_audio = bytearray()
        processing = False
        last_transcription = ""
        silence_frames = 0
        max_silence_frames = 5  # Number of silent chunks before processing
        silence_threshold = 0.01
        
        # Track barge-in detection
        barge_in_detected = False
        
        # Timing stats
        start_time = time.time()
        
        try:
            # Initialize speech recognizer
            await self.speech_recognizer.start_streaming()
            
            # Define result collecting callback
            results = []
            
            async def result_callback(result):
                results.append(result)
                
                # Check for barge-in
                nonlocal barge_in_detected
                if self.using_google and hasattr(self.speech_recognizer, 'is_barge_in_detected'):
                    if self.speech_recognizer.is_barge_in_detected():
                        barge_in_detected = True
                        logger.info("Barge-in detected during streaming")
            
            # Process incoming audio chunks
            async for audio_chunk in audio_chunk_generator:
                # Add to accumulated audio
                if isinstance(audio_chunk, bytes):
                    audio_chunk = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # Check for speech activity
                is_silence = np.mean(np.abs(audio_chunk)) < silence_threshold
                
                if is_silence:
                    silence_frames += 1
                else:
                    silence_frames = 0
                
                # Process the audio chunk
                if self.using_google:
                    # Convert for Google Cloud
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                    result = await self.speech_recognizer.process_audio_chunk(
                        audio_bytes, callback=result_callback
                    )
                else:
                    # Generic approach
                    await self.speech_recognizer.process_audio_chunk(
                        audio_chunk=audio_chunk,
                        callback=result_callback
                    )
                
                # If we have enough silence frames and it's time to process
                if silence_frames >= max_silence_frames and not processing:
                    # Clear results for next utterance
                    results.clear()
                    
                    # Get final transcription
                    if self.using_google:
                        # For Google Cloud
                        transcription, _ = await self.speech_recognizer.stop_streaming()
                        await self.speech_recognizer.start_streaming()
                    else:
                        # Generic approach
                        transcription, _ = await self.speech_recognizer.stop_streaming()
                        
                        # Reset for next utterance
                        if hasattr(self.speech_recognizer, 'start_streaming'):
                            self.speech_recognizer.start_streaming()
                    
                    # Clean up and validate
                    transcription = self.stt_helper.cleanup_transcription(transcription)
                    
                    # Check for barge-in
                    if barge_in_detected:
                        yield {
                            "status": "barge_in",
                            "transcription": transcription
                        }
                        barge_in_detected = False
                    
                    if (transcription and 
                        len(transcription.split()) >= MIN_VALID_WORDS and 
                        transcription != last_transcription):
                        # Process response
                        processing = True
                        try:
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
                                
                                # Update last transcription
                                last_transcription = transcription
                        finally:
                            processing = False
                    
                    # Reset silence counter
                    silence_frames = 0
            
            # Process any final audio
            if self.using_google:
                final_transcription, _ = await self.speech_recognizer.stop_streaming()
            else:
                final_transcription, _ = await self.speech_recognizer.stop_streaming()
                
            final_transcription = self.stt_helper.cleanup_transcription(final_transcription)
            
            if (final_transcription and 
                len(final_transcription.split()) >= MIN_VALID_WORDS and 
                final_transcription != last_transcription):
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
            logger.error(f"Error in real-time stream processing: {e}", exc_info=True)
            yield {
                "status": "error",
                "error": str(e),
                "total_time": time.time() - start_time
            }
        finally:
            # Ensure speech recognizer is properly closed
            try:
                await self.speech_recognizer.stop_streaming()
            except:
                pass