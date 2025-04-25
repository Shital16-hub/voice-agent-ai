"""
Knowledge Base integration module for Voice AI Agent.

This module provides classes and functions for integrating knowledge base
capabilities with the Voice AI Agent system.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List

from knowledge_base.conversation_manager import ConversationManager, ConversationState
from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.utils.cache_utils import StreamingResponseCache

logger = logging.getLogger(__name__)

class KnowledgeBaseIntegration:
    """
    Knowledge Base integration for Voice AI Agent.
    
    Provides an abstraction layer for knowledge base functionality,
    handling query processing and response generation.
    """
    
    def __init__(
        self,
        query_engine: QueryEngine,
        conversation_manager: ConversationManager,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        use_cache: bool = True
    ):
        """
        Initialize the Knowledge Base integration.
        
        Args:
            query_engine: Initialized QueryEngine instance
            conversation_manager: Initialized ConversationManager instance
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response generation
            use_cache: Whether to use response caching
        """
        self.query_engine = query_engine
        self.conversation_manager = conversation_manager
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.initialized = True
        self.use_cache = use_cache
        
        # Initialize response cache
        if self.use_cache:
            self.response_cache = StreamingResponseCache()
    
    async def query(self, text: str, include_context: bool = False) -> Dict[str, Any]:
        """
        Query the knowledge base and generate a response, with caching for improved latency.
        
        Args:
            text: Query text
            include_context: Whether to include context in the response
            
        Returns:
            Dictionary with query response information
        """
        if not self.initialized:
            logger.error("Knowledge Base integration not properly initialized")
            return {"error": "Knowledge Base integration not initialized"}
        
        # Reset conversation manager state if needed
        if self.conversation_manager.current_state != ConversationState.WAITING_FOR_QUERY:
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Track timing
        start_time = time.time()
        
        # Check cache first for fast responses
        if self.use_cache:
            cached_result = self.response_cache.get(text)
            if cached_result:
                logger.info(f"Cache hit for query: '{text}'")
                # Add timing information
                cached_result["total_time"] = time.time() - start_time
                cached_result["cache_hit"] = True
                return cached_result
        
        try:
            # Use a timeout for the entire process to ensure responsiveness
            try:
                # Wrap the query process in a timeout to ensure it doesn't take too long
                # Instead of asyncio.timeout, use asyncio.wait_for
                retrieval_task = asyncio.create_task(self.query_engine.retrieve_with_sources(text))
                
                # Get relevant context with timeout
                retrieval_results = await asyncio.wait_for(retrieval_task, timeout=55.0)
                retrieval_start = time.time()
                results = retrieval_results.get("results", [])
                context = self.query_engine.format_retrieved_context(results)
                retrieval_time = time.time() - retrieval_start
                
                # Generate response with timeout
                llm_start = time.time()
                query_task = asyncio.create_task(self.query_engine.query(text))
                direct_result = await asyncio.wait_for(query_task, timeout=55.0)
                response = direct_result.get("response", "")
                llm_time = time.time() - llm_start
                
            except asyncio.TimeoutError:
                logger.warning(f"Knowledge base query timed out for: '{text}'")
                # Provide a reasonable fallback response
                response = "I'm processing your question, but it's taking longer than expected. Could you please rephrase your question or be more specific?"
                retrieval_time = time.time() - start_time
                llm_time = 0.0
                
                # Try to still include any retrieved results we might have
                if not 'context' in locals() or not context:
                    context = ""
                    if 'results' in locals() and isinstance(results, list) and results:
                        context = f"I found some relevant information but need more time to process it fully: {results[0].get('text', '')[:100]}..."
            
            # Prepare the result
            result = {
                "query": text,
                "response": response,
                "total_time": time.time() - start_time,
                "retrieval_time": retrieval_time,
                "llm_time": llm_time
            }
            
            # Include context if requested
            if include_context:
                result["context"] = context
                result["sources"] = []
                
                # Extract source information
                if 'results' in locals() and isinstance(results, list):
                    for i, doc in enumerate(results):
                        metadata = doc.get("metadata", {})
                        source = metadata.get("source", f"Source {i+1}")
                        result["sources"].append({
                            "id": i,
                            "source": source,
                            "metadata": metadata
                        })
            
            # Store in cache for future queries
            if self.use_cache:
                self.response_cache.set(text, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Knowledge Base query: {e}")
            return {
                "error": str(e),
                "query": text,
                "total_time": time.time() - start_time
            }
    
    async def query_streaming(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Query the knowledge base with streaming response generation.
        
        Args:
            text: Query text
            
        Yields:
            Response chunks
        """
        if not self.initialized:
            logger.error("Knowledge Base integration not properly initialized")
            yield {"error": "Knowledge Base integration not initialized", "done": True}
            return
        
        # Reset conversation manager state if needed
        if self.conversation_manager.current_state != ConversationState.WAITING_FOR_QUERY:
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Check cache for fast responses
        if self.use_cache:
            cached_result = self.response_cache.get(text)
            if cached_result and "response" in cached_result:
                # For cached responses, yield the entire response as a single chunk
                yield {
                    "chunk": cached_result["response"],
                    "done": False,
                    "cache_hit": True
                }
                yield {
                    "chunk": "",
                    "full_response": cached_result["response"],
                    "done": True,
                    "cache_hit": True,
                    "sources": cached_result.get("sources", [])
                }
                return
        
        try:
            # Use the query engine's streaming capability
            async for chunk in self.query_engine.query_with_streaming(text):
                yield chunk
        except Exception as e:
            logger.error(f"Error in streaming Knowledge Base query: {e}")
            yield {
                "error": str(e),
                "done": True
            }
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        if self.conversation_manager:
            self.conversation_manager.reset()
            # Always set to WAITING_FOR_QUERY for voice interaction
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation turns
        """
        if not self.conversation_manager:
            return []
        
        return self.conversation_manager.get_history()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        if not self.query_engine:
            return {"error": "Query engine not initialized"}
        
        # Get stats from query engine
        kb_stats = await self.query_engine.get_stats()
        
        # Add integration-specific stats
        stats = {
            "kb_stats": kb_stats,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add cache stats if enabled
        if self.use_cache:
            stats["cache_stats"] = self.response_cache.get_stats()
        
        # Add conversation stats if available
        if self.conversation_manager and self.conversation_manager.history:
            conversation_stats = {
                "total_turns": len(self.conversation_manager.history),
                "current_state": self.conversation_manager.current_state
            }
            stats["conversation"] = conversation_stats
        
        return stats