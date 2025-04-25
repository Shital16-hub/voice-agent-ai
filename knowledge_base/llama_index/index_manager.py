"""
Index management for LlamaIndex.
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union

from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument

from knowledge_base.config import get_vector_db_config
from knowledge_base.llama_index.schema import Document
from knowledge_base.llama_index.embedding_setup import get_embedding_model

logger = logging.getLogger(__name__)

class IndexManager:
    """
    Manage vector indexes for document storage and retrieval.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_local_storage: bool = True,
        storage_dir: Optional[str] = None,
        embed_model: Optional[Any] = None
    ):
        """
        Initialize IndexManager.
        
        Args:
            config: Optional configuration dictionary
            use_local_storage: Whether to use persistent storage
            storage_dir: Directory for persistent storage
            embed_model: Optional embedding model to use
        """
        self.config = config or get_vector_db_config()
        self.collection_name = self.config["collection_name"]
        self.vector_size = self.config["vector_size"]
        
        # Set storage directory
        self.use_local_storage = use_local_storage
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "chroma_db")
        
        # Create storage directory if it doesn't exist
        if self.use_local_storage:
            os.makedirs(self.storage_dir, exist_ok=True)
            logger.info(f"Using persistent storage at: {os.path.abspath(self.storage_dir)}")
        
        # Store initialization state
        self.is_initialized = False
        self.vector_store = None
        self.index = None
        self._embed_model = embed_model
        
        logger.info(f"Initialized IndexManager with collection: {self.collection_name}")
    
    async def init(self):
        """Initialize the index and vector store."""
        if self.is_initialized:
            return
        
        try:
            # Import necessary components
            import chromadb
            from llama_index.core import Settings
            
            # Set up the embedding model
            embed_model = self._embed_model
            if embed_model is None:
                embed_model = get_embedding_model()
            
            # Configure global settings
            Settings.embed_model = embed_model
            Settings.llm = None  # Explicitly disable LLM usage
            
            # Create ChromaDB client
            if self.use_local_storage:
                # Persistent client
                chroma_client = chromadb.PersistentClient(path=self.storage_dir)
                logger.info(f"Connected to persistent ChromaDB at {self.storage_dir}")
            else:
                # In-memory client
                chroma_client = chromadb.EphemeralClient()
                logger.info("Using in-memory ChromaDB")
            
            # Get or create collection
            collection = chroma_client.get_or_create_collection(self.collection_name)
            
            # Create vector store
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create vector store index
            self.index = VectorStoreIndex.from_documents(
                documents=[],  # Empty initially
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            # Get collection stats
            doc_count = collection.count()
            logger.info(f"Connected to collection '{self.collection_name}' with {doc_count} documents")
            
            self.is_initialized = True
            
        except ImportError as e:
            logger.error(f"Error importing required packages: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing IndexManager: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the index.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        if not self.is_initialized:
            await self.init()
        
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []
        
        try:
            # Convert to LlamaIndex documents
            llama_docs = [doc.to_llama_index_document() for doc in documents]
            
            # Add to index
            doc_ids = []
            for doc in llama_docs:
                self.index.insert(doc)
                doc_ids.append(doc.id_)
            
            logger.info(f"Added {len(doc_ids)} documents to index")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents from the index.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Delete from index
            for doc_id in doc_ids:
                self.index.delete(doc_id)
            
            logger.info(f"Deleted {len(doc_ids)} documents from index")
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
    
    async def count_documents(self) -> int:
        """
        Count documents in the index.
        
        Returns:
            Number of documents
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Get collection
            doc_count = self.vector_store.client.count()
            return doc_count
            
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    async def reset_index(self) -> bool:
        """
        Reset the index by deleting and recreating it.
        
        Returns:
            True if successful
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Delete collection
            self.vector_store.client.delete_collection(self.collection_name)
            
            # Reinitialize
            self.is_initialized = False
            await self.init()
            
            logger.info(f"Reset index collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False