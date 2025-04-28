"""
Voice AI Agent main class that coordinates all components with improved noise handling.
"""
import os
import logging
import asyncio
from typing import Optional, Dict, Any

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, PARAMETER_PRESETS
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import DeepgramTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class that coordinates all components with noise handling improvements."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        whisper_model_path: str = 'models/base.en',
        llm_temperature: float = 0.7,
        # Add noise handling parameters
        whisper_initial_prompt: str = "This is a clear business conversation in English. Transcribe the exact words spoken, ignoring background noise.",
        whisper_temperature: float = 0.0,
        whisper_no_context: bool = True,
        whisper_preset: Optional[str] = "default"
    ):
        """Initialize the Voice AI Agent with improved noise handling configuration."""
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.whisper_model_path = whisper_model_path
        self.llm_temperature = llm_temperature
        
        # Store noise handling parameters
        self.whisper_initial_prompt = whisper_initial_prompt
        self.whisper_temperature = whisper_temperature
        self.whisper_no_context = whisper_no_context
        self.whisper_preset = whisper_preset
        
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
    async def init(self):
        """Initialize all components with noise handling optimizations."""
        logger.info("Initializing Voice AI Agent components with noise optimization...")
        
        # Initialize speech recognizer with improved noise handling parameters
        self.speech_recognizer = StreamingWhisperASR(
            model_path=self.whisper_model_path,
            language="en",
            n_threads=4,
            chunk_size_ms=2000,
            vad_enabled=True,
            single_segment=True,
            # Use our noise handling parameters
            temperature=self.whisper_temperature,
            initial_prompt=self.whisper_initial_prompt,
            no_context=self.whisper_no_context,
            # Use the preset if provided
            preset=self.whisper_preset
        )
        
        # Initialize STT integration
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language="en"
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
        self.tts_client = DeepgramTTS()
        
        logger.info("Voice AI Agent initialization complete with noise handling optimizations")
        
    async def process_audio(self, audio_data, callback=None):
        """
        Process audio data with noise handling optimizations.
        
        Args:
            audio_data: Audio data as numpy array
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Use STT integration for processing with noise handling
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(transcription)
            
            return {
                "transcription": transcription,
                "response": response.get("response", ""),
                "status": "success"
            }
        else:
            logger.info("Invalid or empty transcription")
            return {
                "status": "invalid_transcription",
                "transcription": result.get("transcription", ""),
                "error": "No valid speech detected"
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
        # Any cleanup needed here
        
        # Example: If there's an active conversation
        if self.conversation_manager:
            self.conversation_manager.reset()