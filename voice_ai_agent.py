"""
Enhanced Voice AI Agent main class that coordinates all components with improved
speech/noise discrimination and Google Cloud Speech-to-Text integration for telephony applications.
This version makes the AudioPreprocessor optional and uses direct processing methods.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np

# Import the enhanced audio preprocessor
from telephony.audio_preprocessor import AudioPreprocessor

# Google STT imports
from speech_to_text.google_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import GoogleCloudTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """
    Enhanced Voice AI Agent class with optional AudioPreprocessor component
    that uses Google Cloud Speech-to-Text for superior speech recognition.
    """
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        credentials_path: Optional[str] = None,
        llm_temperature: float = 0.7,
        enable_debug: bool = False,
        skip_audio_preprocessor: bool = False,  # Add option to skip audio preprocessor
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with enhanced speech processing.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            credentials_path: Path to Google Cloud credentials JSON
            llm_temperature: LLM temperature for response generation
            enable_debug: Enable detailed debug logging
            skip_audio_preprocessor: Whether to skip the audio preprocessor
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.llm_temperature = llm_temperature
        self.enable_debug = enable_debug
        self.skip_audio_preprocessor = skip_audio_preprocessor
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        
        # Optimized telephony keywords
        self.stt_keywords = kwargs.get('keywords', [
            'price', 'plan', 'cost', 'subscription', 'service', 'features',
            'support', 'help', 'agent', 'assistant', 'voice'
        ])
        
        # Additional STT parameters
        self.stt_model = kwargs.get('stt_model', 'phone_call')
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Create the dedicated audio preprocessor only if not skipped
        if not skip_audio_preprocessor:
            self.audio_preprocessor = AudioPreprocessor(
                sample_rate=16000,
                enable_barge_in=True,
                barge_in_threshold=0.03,  # Decreased from 0.055
                min_speech_frames_for_barge_in=6,  # Decreased from 12
                barge_in_cooldown_ms=1000,  # Decreased from 2000
                enable_debug=self.enable_debug
            )
            logger.info("Initialized Voice AI Agent with audio preprocessor")
        else:
            self.audio_preprocessor = None
            logger.info("Initialized Voice AI Agent without audio preprocessor")
                
    async def init(self):
        """Initialize all components with enhanced speech processing using Google Cloud."""
        logger.info("Initializing Voice AI Agent components with enhanced speech processing...")
        
        try:
            # Check if Google Cloud credentials are available
            if not self.credentials_path:
                raise ValueError("Google Cloud credentials are required for Speech-to-Text")

            # Initialize speech recognizer with Google Cloud
            try:
                # Create an optimized Google Cloud client with telephony-specific settings
                self.speech_recognizer = GoogleCloudStreamingSTT(
                    credentials_path=self.credentials_path,
                    language_code=self.stt_language,
                    sample_rate=8000,  # 8kHz for telephony
                    encoding="LINEAR16",
                    channels=1,
                    interim_results=True,  # Enable interim results for responsiveness
                    model=self.stt_model  # Use phone_call model
                )
                logger.info("Successfully initialized Google Cloud Speech-to-Text")
                
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud Speech-to-Text: {e}")
                raise
            
            # Initialize STT integration with better validation thresholds
            self.stt_integration = STTIntegration(
                speech_recognizer=self.speech_recognizer,
                language=self.stt_language
            )
            
            # Initialize document store and index manager
            doc_store = DocumentStore()
            index_manager = IndexManager(storage_dir=self.storage_dir)
            await index_manager.init()
            
            # Initialize query engine with improved configuration
            self.query_engine = QueryEngine(
                index_manager=index_manager, 
                llm_model_name=self.model_name,
                llm_temperature=self.llm_temperature
            )
            await self.query_engine.init()
            
            # Initialize conversation manager with optimized parameters
            self.conversation_manager = ConversationManager(
                query_engine=self.query_engine,
                llm_model_name=self.model_name,
                llm_temperature=self.llm_temperature,
                # Skip greeting for better telephone experience
                skip_greeting=True
            )
            await self.conversation_manager.init()
            
            # Initialize TTS client
            self.tts_client = GoogleCloudTTS(credentials_path=self.credentials_path)
            
            logger.info("Voice AI Agent initialization complete with enhanced speech processing")
            
        except Exception as e:
            logger.error(f"Error initializing Voice AI Agent: {e}", exc_info=True)
            raise
    
    def process_audio_with_enhanced_preprocessing(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio with the dedicated AudioPreprocessor or direct processing.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        if self.skip_audio_preprocessor or self.audio_preprocessor is None:
            # Use direct processing instead of AudioPreprocessor
            return self._direct_audio_processing(audio_data)
        else:
            # Use the dedicated audio preprocessor
            return self.audio_preprocessor.process_audio(audio_data)
    
    def _direct_audio_processing(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Direct audio processing without using AudioPreprocessor.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        try:
            # Simple noise gate
            noise_threshold = 0.01
            audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)
            
            # Simple normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95
                
            return audio_data
        except Exception as e:
            logger.error(f"Error in direct audio processing: {e}")
            return audio_data  # Return original if processing fails
    
    def detect_speech_with_enhanced_processor(self, audio_data: np.ndarray) -> bool:
        """
        Detect speech using either AudioPreprocessor or direct method.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if speech is detected
        """
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        if self.skip_audio_preprocessor or self.audio_preprocessor is None:
            # Use direct speech detection
            return self._direct_speech_detection(audio_data)
        else:
            # Use the dedicated audio preprocessor's speech detection
            return self.audio_preprocessor.contains_speech(audio_data)
    
    def _direct_speech_detection(self, audio_data: np.ndarray) -> bool:
        """
        Direct speech detection without AudioPreprocessor.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if speech detected
        """
        try:
            # Calculate energy of the audio
            energy = np.mean(np.abs(audio_data))
            
            # Use a simple threshold
            speech_threshold = 0.02  # Very low threshold to catch most speech
            
            return energy > speech_threshold
        except Exception as e:
            logger.error(f"Error in direct speech detection: {e}")
            return True  # Default to assuming speech is present
    
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with improved speech/noise discrimination.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not hasattr(self, 'speech_recognizer') or not self.speech_recognizer:
            raise RuntimeError("Voice AI Agent not initialized")
        
        start_time = time.time()
        
        # Process audio for better speech recognition
        if isinstance(audio_data, np.ndarray):
            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Apply enhanced preprocessing
            audio_data = self.process_audio_with_enhanced_preprocessing(audio_data)
            
            # Check if audio contains actual speech
            contains_speech = self.detect_speech_with_enhanced_processor(audio_data)
            if not contains_speech:
                logger.info("No speech detected in audio, skipping processing")
                return {
                    "status": "no_speech",
                    "transcription": "",
                    "error": "No speech detected",
                    "processing_time": time.time() - start_time
                }
        
        # Use STT integration for processing with more lenient validation
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Add additional validation for noise filtering - reduced filtering
        transcription = result.get("transcription", "")
        
        # Filter out only obviously noise-only transcriptions
        extreme_noise_keywords = ["static", "(", ")", "[", "]"]
        if any(keyword in transcription.lower() for keyword in extreme_noise_keywords) and len(transcription.split()) <= 1:
            logger.info(f"Filtered out extreme noise transcription: '{transcription}'")
            return {
                "status": "filtered_noise",
                "transcription": transcription,
                "filtered": True,
                "processing_time": time.time() - start_time
            }
        
        # More lenient processing - only require 2 words minimum
        if result.get("is_valid", False) and transcription and len(transcription.split()) >= 2:
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(transcription)
            
            return {
                "transcription": transcription,
                "response": response.get("response", ""),
                "status": "success",
                "processing_time": time.time() - start_time
            }
        else:
            logger.info(f"Transcription too short or invalid: '{transcription}'")
            return {
                "status": "invalid_transcription",
                "transcription": transcription,
                "error": "No valid speech detected or transcription too short",
                "processing_time": time.time() - start_time
            }
    
    async def process_streaming_audio(
        self,
        audio_stream,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process streaming audio with real-time response and enhanced speech processing.
        
        Args:
            audio_stream: Async iterator of audio chunks
            result_callback: Callback for streaming results
            
        Returns:
            Final processing stats
        """
        if not hasattr(self, 'speech_recognizer') or not self.speech_recognizer:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Track stats
        start_time = time.time()
        chunks_processed = 0
        results_count = 0
        
        # Start streaming session
        if hasattr(self.speech_recognizer, 'start_streaming'):
            await self.speech_recognizer.start_streaming()
        
        try:
            # Process each audio chunk
            async for chunk in audio_stream:
                chunks_processed += 1
                
                # Process audio for better recognition if it's numpy array
                if isinstance(chunk, np.ndarray):
                    # Ensure float32 format
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    
                    # Apply enhanced preprocessing with reduced complexity
                    chunk = self.process_audio_with_enhanced_preprocessing(chunk)
                    
                    # Skip further processing if no speech detected - less strict now
                    # Only skip every 5th chunk to ensure we don't miss speech onset
                    if not self.detect_speech_with_enhanced_processor(chunk) and chunks_processed % 5 != 0:
                        continue
                
                # Convert to bytes for STT if needed
                if isinstance(chunk, np.ndarray):
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = chunk
                    
                # Process through STT with reduced filtering
                async def process_result(result):
                    # Process all results, not just final ones
                    is_final = getattr(result, 'is_final', True)
                    
                    # Clean up transcription more leniently
                    transcription = self.stt_integration.cleanup_transcription(result.text)
                    
                    # Process all non-empty transcriptions that have at least 2 words
                    if transcription and len(transcription.split()) >= 2:
                        # Get response from conversation manager
                        response = await self.conversation_manager.handle_user_input(transcription)
                        
                        # Format result
                        result_data = {
                            "transcription": transcription,
                            "response": response.get("response", ""),
                            "confidence": getattr(result, 'confidence', 1.0),
                            "is_final": is_final
                        }
                        
                        nonlocal results_count
                        results_count += 1
                        
                        # Call callback if provided
                        if result_callback:
                            await result_callback(result_data)
                
                # Process chunk
                if hasattr(self.speech_recognizer, 'process_audio_chunk'):
                    await self.speech_recognizer.process_audio_chunk(audio_bytes, process_result)
                
            # Stop streaming session
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                await self.speech_recognizer.stop_streaming()
            
            # Return stats
            return {
                "status": "complete",
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming audio: {e}")
            
            # Stop streaming session
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                await self.speech_recognizer.stop_streaming()
            
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return (self.speech_recognizer is not None and 
                self.conversation_manager is not None and 
                self.query_engine is not None)
                
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Close STT streaming session if active
        if self.speech_recognizer:
            if hasattr(self.speech_recognizer, 'is_streaming') and getattr(self.speech_recognizer, 'is_streaming', False):
                if hasattr(self.speech_recognizer, 'stop_streaming'):
                    await self.speech_recognizer.stop_streaming()
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()
            
        # Reset audio preprocessor state if it exists
        if self.audio_preprocessor:
            self.audio_preprocessor.reset()