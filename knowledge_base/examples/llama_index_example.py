#!/usr/bin/env python3
"""
Example script for using LlamaIndex integration.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def index_documents(directory: str, storage_dir: str = './storage') -> int:
    """
    Index documents from a directory.
    
    Args:
        directory: Directory containing documents
        storage_dir: Directory for persistent storage
        
    Returns:
        Number of documents indexed
    """
    logger.info(f"Indexing documents from {directory}")
    
    # Initialize document store
    doc_store = DocumentStore()
    
    # Load and process documents
    documents = doc_store.load_documents_from_directory(directory)
    
    # Initialize index manager
    index_manager = IndexManager(storage_dir=storage_dir)
    await index_manager.init()
    
    # Add documents to index
    doc_ids = await index_manager.add_documents(documents)
    
    # Get stats
    doc_count = await index_manager.count_documents()
    logger.info(f"Indexed {len(doc_ids)} new documents, total count: {doc_count}")
    
    return len(doc_ids)

async def query_knowledge_base():
    """Interactive query example."""
    # Initialize components
    index_manager = IndexManager()
    await index_manager.init()
    
    query_engine = QueryEngine(index_manager=index_manager)
    await query_engine.init()
    
    # Get stats
    stats = await query_engine.get_stats()
    doc_count = stats["document_count"]
    
    print("\n" + "="*50)
    print(f"Knowledge Base Query Example")
    print(f"Documents in knowledge base: {doc_count}")
    print("Type 'exit' or 'quit' to end the session")
    print("="*50 + "\n")
    
    if doc_count == 0:
        print("Warning: No documents found in knowledge base.")
        should_index = input("Would you like to index some documents? (y/n): ")
        if should_index.lower() == 'y':
            directory = input("Enter the directory path containing documents: ")
            await index_documents(directory)
    
    while True:
        # Get query
        query = input("\nEnter your query: ")
        
        # Check for exit
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Process query
        start_time = asyncio.get_event_loop().time()
        results = await query_engine.retrieve_with_sources(query)
        query_time = asyncio.get_event_loop().time() - start_time
        
        # Display results
        print(f"\nRetrieved {len(results['results'])} results in {query_time:.3f}s:")
        for i, doc in enumerate(results['results']):
            print(f"\n--- Result {i+1} ---")
            print(f"Score: {doc.get('score', 0):.4f}")
            source = doc["metadata"].get("source", "Unknown")
            print(f"Source: {source}")
            
            # Print truncated text
            text = doc["text"]
            if len(text) > 500:
                text = text[:497] + "..."
            print(f"Content: {text}")
        
        # Try a direct query
        print("\nGenerating a response from the knowledge base...")
        response = await query_engine.query(query)
        
        print(f"\nResponse: {response['response']}")
        print(f"Based on {len(response['sources'])} sources")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaIndex Knowledge Base Example")
    parser.add_argument('--index', action='store_true', help='Index documents')
    parser.add_argument('--query', action='store_true', help='Query knowledge base')
    parser.add_argument('--directory', type=str, help='Directory containing documents to index')
    parser.add_argument('--storage', type=str, default='./storage', help='Storage directory for index')
    
    args = parser.parse_args()
    
    if args.index:
        if not args.directory:
            print("Error: --directory is required when using --index")
            return 1
        
        await index_documents(args.directory, args.storage)
    
    if args.query or (not args.index):
        await query_knowledge_base()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)