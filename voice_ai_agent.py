"""
Enhanced Voice AI Agent main class that coordinates all components with improved
speech/noise discrimination and Google Cloud Speech-to-Text integration for telephony applications.
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
    Enhanced Voice AI Agent class that integrates the AudioPreprocessor component
    and uses Google Cloud Speech-to-Text for superior speech/noise discrimination.
    """
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        credentials_path: Optional[str] = None,
        llm_temperature: float = 0.7,
        enable_debug: bool = False,
        **kwargs
    ):
        
        """
        Initialize the Voice AI Agent with minimal speech processing.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            credentials_path: Path to Google Cloud credentials JSON
            llm_temperature: LLM temperature for response generation
            enable_debug: Enable detailed debug logging
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.llm_temperature = llm_temperature
        self.enable_debug = enable_debug
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        
        # Optimized telephony keywords
        self.stt_keywords = kwargs.get('keywords', [
            'price', 'plan', 'cost', 'subscription', 'service', 'features',
            'support', 'help', 'agent', 'assistant', 'voice'
        ])
        
        # Additional STT parameters
        self.stt_model = kwargs.get('stt_model', 'phone_call')
        
        # Skip using the enhanced audio preprocessor - use a minimal version
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            enable_barge_in=True,
            # Set very lenient thresholds
            barge_in_threshold=0.03,  # Decreased from 0.055
            min_speech_frames_for_barge_in=6,  # Decreased from 12
            barge_in_cooldown_ms=1000,  # Decreased from 2000
            enable_debug=False  # Disable debug logging
        )
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        logger.info("Initialized Voice AI Agent with minimal audio preprocessing")
                
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
        Process audio with the dedicated AudioPreprocessor for superior speech/noise discrimination.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Use the dedicated audio preprocessor
        return self.audio_preprocessor.process_audio(audio_data)
    
    def detect_speech_with_enhanced_processor(self, audio_data: np.ndarray) -> bool:
        """
        Detect speech using the enhanced AudioPreprocessor.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if speech is detected
        """
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Use the dedicated audio preprocessor's speech detection
        return self.audio_preprocessor.contains_speech(audio_data)
    
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with enhanced speech/noise discrimination.
        
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
        
        # Use STT integration for processing
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Add additional validation for noise filtering
        transcription = result.get("transcription", "")
        
        # Filter out noise-only transcriptions
        noise_keywords = ["click", "keyboard", "tongue", "scoff", "static", "*", "(", ")"]
        if any(keyword in transcription.lower() for keyword in noise_keywords):
            logger.info(f"Filtered out noise transcription: '{transcription}'")
            return {
                "status": "filtered_noise",
                "transcription": transcription,
                "filtered": True,
                "processing_time": time.time() - start_time
            }
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and transcription and len(transcription.split()) >= 3:
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
            logger.info(f"Invalid or too short transcription: '{transcription}'")
            return {
                "status": "invalid_transcription",
                "transcription": transcription,
                "error": "No valid speech detected",
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
                    
                    # Apply enhanced preprocessing
                    chunk = self.process_audio_with_enhanced_preprocessing(chunk)
                    
                    # Skip further processing if no speech detected
                    if not self.detect_speech_with_enhanced_processor(chunk) and chunks_processed % 5 != 0:
                        # Only skip some chunks to ensure we don't miss speech onset
                        continue
                
                # Convert to bytes for STT if needed
                if isinstance(chunk, np.ndarray):
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = chunk
                    
                # Process through STT with noise filtering
                async def process_result(result):
                    # Only handle final results
                    if getattr(result, 'is_final', True):
                        # Clean up transcription
                        transcription = self.stt_integration.cleanup_transcription(result.text)
                        
                        # Filter out noise transcriptions
                        noise_keywords = ["click", "keyboard", "tongue", "scoff", "static", "*", "(", ")"]
                        if any(keyword in transcription.lower() for keyword in noise_keywords):
                            logger.info(f"Filtered out noise transcription: '{transcription}'")
                            return
                        
                        # Process if valid and has minimum words
                        if transcription and self.stt_integration.is_valid_transcription(transcription) and len(transcription.split()) >= 3:
                            # Get response from conversation manager
                            response = await self.conversation_manager.handle_user_input(transcription)
                            
                            # Format result
                            result_data = {
                                "transcription": transcription,
                                "response": response.get("response", ""),
                                "confidence": getattr(result, 'confidence', 1.0),
                                "is_final": True
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
            
        # Reset audio preprocessor state
        self.audio_preprocessor.reset()