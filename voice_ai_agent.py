"""
Voice AI Agent main class that coordinates all components.
"""
import os
import logging
import asyncio
from typing import Optional, Dict, Any

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import DeepgramTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class that coordinates all components."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        whisper_model_path: str = 'models/base.en',
        llm_temperature: float = 0.7
    ):
        """Initialize the Voice AI Agent."""
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.whisper_model_path = whisper_model_path
        self.llm_temperature = llm_temperature
        
        self.speech_recognizer = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
    async def init(self):
        """Initialize all components."""
        logger.info("Initializing Voice AI Agent components...")
        
        # Initialize speech recognizer
        self.speech_recognizer = StreamingWhisperASR(
            model_path=self.whisper_model_path,
            language="en",
            n_threads=4,
            chunk_size_ms=2000,
            vad_enabled=True,
            single_segment=True,
            temperature=0.0
        )
        
        # Initialize document store and index manager
        doc_store = DocumentStore()
        index_manager = IndexManager(storage_dir=self.storage_dir)
        await index_manager.init()
        
        # Initialize query engine
        self.query_engine = QueryEngine(index_manager=index_manager)
        await self.query_engine.init()
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature
        )
        await self.conversation_manager.init()
        
        # Initialize TTS client
        self.tts_client = DeepgramTTS()
        
        logger.info("Voice AI Agent initialization complete")