"""
Voice AI Agent main class with Google Cloud STT integration.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np

from integration.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import GoogleCloudTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """
    Voice AI Agent class that integrates Google Cloud STT for superior
    speech recognition in telephony applications.
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
        Initialize the Voice AI Agent with Google Cloud STT integration.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            credentials_path: Google Cloud credentials path
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
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        logger.info(f"Initialized Voice AI Agent with Google Cloud STT integration")
                
    async def init(self):
        """Initialize all components with Google Cloud STT integration."""
        logger.info("Initializing Voice AI Agent components...")
        
        try:
            # Initialize Google Cloud STT integration
            from speech_to_text.google_stt import GoogleCloudSTT
            
            self.speech_recognizer = GoogleCloudSTT(
                credentials_path=self.credentials_path,
                language_code=self.stt_language,
                model="phone_call",
                use_enhanced=True
            )
            
            # Initialize STT integration
            self.stt_integration = STTIntegration(
                stt_client=self.speech_recognizer,
                language=self.stt_language
            )
            
            # Initialize document store and index manager
            doc_store = DocumentStore()
            index_manager = IndexManager(storage_dir=self.storage_dir)
            await index_manager.init()
            
            # Initialize query engine
            self.query_engine = QueryEngine(
                index_manager=index_manager, 
                llm_model_name=self.model_name,
                llm_temperature=self.llm_temperature
            )
            await self.query_engine.init()
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(
                query_engine=self.query_engine,
                llm_model_name=self.model_name,
                llm_temperature=self.llm_temperature,
                skip_greeting=True
            )
            await self.conversation_manager.init()
            
            # Initialize TTS client
            self.tts_client = GoogleCloudTTS()
            
            logger.info("Voice AI Agent initialization complete with Google Cloud STT")
            
        except Exception as e:
            logger.error(f"Error initializing Voice AI Agent: {e}", exc_info=True)
            raise
    
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with Google Cloud STT.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not hasattr(self, 'speech_recognizer') or not self.speech_recognizer:
            raise RuntimeError("Voice AI Agent not initialized")
        
        start_time = time.time()
        
        # Process audio through STT integration
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Get transcription
        transcription = result.get("transcription", "")
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and transcription:
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
            logger.info(f"Invalid or empty transcription: '{transcription}'")
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
        Process streaming audio with real-time response.
        
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
        await self.stt_integration.start_streaming()
        
        try:
            # Process each audio chunk
            async for chunk in audio_stream:
                chunks_processed += 1
                
                # Convert to bytes for STT if needed
                if isinstance(chunk, np.ndarray):
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = chunk
                    
                # Process through STT
                async def process_result(result):
                    # Only handle valid results
                    if result.get("is_valid", False):
                        transcription = result.get("transcription", "")
                        
                        # Process if valid and has minimum words
                        if transcription:
                            # Get response from conversation manager
                            response = await self.conversation_manager.handle_user_input(transcription)
                            
                            # Format result
                            result_data = {
                                "transcription": transcription,
                                "response": response.get("response", ""),
                                "confidence": result.get("confidence", 1.0),
                                "is_final": result.get("is_final", True)
                            }
                            
                            nonlocal results_count
                            results_count += 1
                            
                            # Call callback if provided
                            if result_callback:
                                await result_callback(result_data)
                
                # Process chunk
                await self.stt_integration.process_stream_chunk(audio_bytes, process_result)
                
            # Stop streaming session
            await self.stt_integration.end_streaming()
            
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
            await self.stt_integration.end_streaming()
            
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
        if self.stt_integration:
            await self.stt_integration.end_streaming()
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()