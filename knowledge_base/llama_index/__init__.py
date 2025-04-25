"""
LlamaIndex integration components for knowledge base.
"""
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.embedding_setup import get_embedding_model
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.schema import Document, DocumentMetadata

__all__ = [
    "DocumentStore",
    "get_embedding_model",
    "IndexManager",
    "QueryEngine",
    "Document",
    "DocumentMetadata"
]