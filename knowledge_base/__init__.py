"""
Knowledge base component for the Voice AI Agent.
"""
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.embedding_setup import get_embedding_model
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.schema import Document, DocumentMetadata
from knowledge_base.conversation_manager import ConversationManager, ConversationState, ConversationTurn

__version__ = "0.2.0"

__all__ = [
    "Document",
    "DocumentMetadata",
    "DocumentStore",
    "get_embedding_model",
    "IndexManager",
    "QueryEngine",
    "ConversationManager",
    "ConversationState",
    "ConversationTurn",
]