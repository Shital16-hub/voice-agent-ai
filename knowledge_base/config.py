"""
Configuration settings for the knowledge base component.
"""
import os
from typing import Dict, Any, List, Optional

# Vector database settings
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "6333"))
VECTOR_DB_GRPC_PORT = int(os.getenv("VECTOR_DB_GRPC_PORT", "6334"))
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "company_knowledge")
VECTOR_DIMENSION = 384  # For sentence-transformers/all-MiniLM-L6-v2

# Embedding model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")  # Smaller model
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # Set to "cuda" for GPU
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))

# Retrieval settings
DEFAULT_RETRIEVE_COUNT = int(os.getenv("DEFAULT_RETRIEVE_COUNT", "3"))
MINIMUM_RELEVANCE_SCORE = float(os.getenv("MINIMUM_RELEVANCE_SCORE", "0.6"))
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "False").lower() == "true"

# Conversation context settings
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "5"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "4096"))

# LlamaIndex settings
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"

# Supported file types
SUPPORTED_DOCUMENT_TYPES = [
    # Text files
    ".txt", ".md", ".csv", ".json",
    
    # Office documents
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    
    # Web content
    ".html", ".htm", ".xml",
]

def get_llama_index_config() -> Dict[str, Any]:
    """
    Get LlamaIndex configuration.
    
    Returns:
        Dictionary with LlamaIndex configuration
    """
    return {
        "persist_dir": PERSIST_DIR,
        "use_gpu": USE_GPU,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": EMBEDDING_DEVICE
    }

def get_document_processor_config() -> Dict[str, Any]:
    """
    Get document processor configuration.
    
    Returns:
        Dictionary with document processor configuration
    """
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_document_size_mb": MAX_DOCUMENT_SIZE_MB,
        "supported_types": SUPPORTED_DOCUMENT_TYPES
    }

def get_embedding_config() -> Dict[str, Any]:
    """
    Get embedding generator configuration.
    
    Returns:
        Dictionary with embedding configuration
    """
    return {
        "model_name": EMBEDDING_MODEL,
        "device": EMBEDDING_DEVICE,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "dimension": VECTOR_DIMENSION
    }

def get_vector_db_config() -> Dict[str, Any]:
    """
    Get vector database configuration.
    
    Returns:
        Dictionary with vector database configuration
    """
    return {
        "host": VECTOR_DB_HOST,
        "port": VECTOR_DB_PORT,
        "grpc_port": VECTOR_DB_GRPC_PORT,
        "collection_name": VECTOR_DB_COLLECTION,
        "vector_size": VECTOR_DIMENSION
    }

def get_retriever_config() -> Dict[str, Any]:
    """
    Get retriever configuration.
    
    Returns:
        Dictionary with retriever configuration
    """
    return {
        "top_k": DEFAULT_RETRIEVE_COUNT,
        "min_score": MINIMUM_RELEVANCE_SCORE,
        "reranking_enabled": RERANKING_ENABLED
    }