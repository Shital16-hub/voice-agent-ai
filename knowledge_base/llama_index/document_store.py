"""
Document processing and handling for LlamaIndex integration.
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator, Union
from pathlib import Path
import hashlib

import llama_index.core.schema as llama_schema
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader, DocxReader, PDFReader
from llama_index.core.ingestion import IngestionPipeline

from knowledge_base.config import get_document_processor_config
from knowledge_base.utils.file_utils import list_documents, check_supported_file
from knowledge_base.llama_index.schema import Document, DocumentMetadata

logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Store and process documents for knowledge base ingestion.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DocumentStore.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_document_processor_config()
        self.chunk_size = self.config["chunk_size"]
        self.chunk_overlap = self.config["chunk_overlap"]
        self.max_document_size = self.config["max_document_size_mb"] * 1024 * 1024
        self.supported_types = self.config["supported_types"]
        
        # Create node parser (document chunker)
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Initialize file readers
        self.readers = {
            ".pdf": PyMuPDFReader(),
            ".docx": DocxReader(),
            ".txt": None,  # Will use simple text reading
            ".md": None,   # Will use simple text reading
        }
        
        logger.info(f"Initialized DocumentStore with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file type is supported
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_types
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load and process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file type is supported
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_document_size:
            raise ValueError(f"File too large: {file_size} bytes "
                           f"(max: {self.max_document_size} bytes)")
        
        # Extract file metadata
        file_metadata = self._extract_file_metadata(file_path)
        
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Use appropriate reader or fallback method
            llama_docs = []
            
            if ext in self.readers and self.readers[ext] is not None:
                # Use LlamaIndex reader
                reader = self.readers[ext]
                llama_docs = reader.load_data(file_path)
            else:
                # Fallback to simple text loading
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                    llama_docs = [llama_schema.Document(text=text, metadata=file_metadata)]
            
            # Split documents into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents(llama_docs)
            
            # Convert to our document format
            documents = []
            for i, node in enumerate(nodes):
                # Create metadata for this chunk
                chunk_metadata = file_metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_count"] = len(nodes)
                
                # Create Document
                doc = Document(
                    text=node.text,
                    metadata=chunk_metadata,
                    doc_id=f"{os.path.basename(file_path)}_{i}"
                )
                documents.append(doc)
            
            logger.info(f"Processed document {file_path}: created {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def load_text(self, text: str, source_name: str = "text_input") -> List[Document]:
        """
        Load and process text directly.
        
        Args:
            text: Text content
            source_name: Name to use as source identifier
            
        Returns:
            List of Document objects
        """
        # Create basic metadata
        metadata = {
            "source": source_name,
            "source_type": "direct_text",
            "file_path": None,
            "file_type": None,
            "file_name": None,
            "created_at": None,
            "modified_at": None
        }
        
        # Create LlamaIndex document
        llama_doc = llama_schema.Document(text=text, metadata=metadata)
        
        # Split into chunks
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])
        
        # Create Document objects
        documents = []
        for i, node in enumerate(nodes):
            # Create metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(nodes)
            
            # Create Document
            doc = Document(
                text=node.text,
                metadata=chunk_metadata,
                doc_id=f"{source_name}_{i}"
            )
            documents.append(doc)
        
        logger.info(f"Processed text input '{source_name}': created {len(documents)} chunks")
        return documents
    
    def load_documents_from_directory(
        self, 
        directory: str, 
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Directory path
            extensions: Optional list of file extensions to include
            max_files: Optional maximum number of files to process
            
        Returns:
            List of Document objects
        """
        # Get list of files
        file_paths = list_documents(directory, extensions=extensions)
        
        if max_files is not None and len(file_paths) > max_files:
            logger.info(f"Limiting to {max_files} files")
            file_paths = file_paths[:max_files]
        
        # Process each file
        all_documents = []
        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Processed {len(file_paths)} files into {len(all_documents)} document chunks")
        return all_documents
    
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with metadata
        """
        file_stat = os.stat(file_path)
        return {
            "source": os.path.basename(file_path),
            "source_type": "file",
            "file_path": os.path.abspath(file_path),
            "file_type": os.path.splitext(file_path)[1].lower(),
            "file_name": os.path.basename(file_path),
            "file_size": file_stat.st_size,
            "created_at": file_stat.st_ctime,
            "modified_at": file_stat.st_mtime
        }