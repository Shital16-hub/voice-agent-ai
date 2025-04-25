"""
Embedding model setup for LlamaIndex.
"""
import logging
from typing import Dict, Optional, Any
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding

from knowledge_base.config import get_embedding_config

logger = logging.getLogger(__name__)

def get_embedding_model(config: Optional[Dict[str, Any]] = None) -> BaseEmbedding:
    """
    Get the embedding model for document indexing and querying.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LlamaIndex embedding model
    """
    # Get embedding configuration
    if config is None:
        config = get_embedding_config()
    
    model_name = config["model_name"]
    device = config["device"]
    
    try:
        logger.info(f"Initializing embedding model: {model_name} on {device}")
        
        # Create the HuggingFace embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            cache_folder=None,  # Use default cache
            embed_batch_size=config.get("batch_size", 32)
        )
        
        logger.info(f"Successfully initialized embedding model: {model_name}")
        return embed_model
        
    except ImportError as e:
        logger.error(f"Error importing required libraries: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise