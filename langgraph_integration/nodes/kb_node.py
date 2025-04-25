"""
Knowledge Base node for the LangGraph-based Voice AI Agent.

This module provides the KB node that processes queries
and generates responses within the LangGraph flow.
"""
import time
import logging
import asyncio
from typing import Dict, Any, AsyncIterator, Optional, List

from integration.kb_integration import KnowledgeBaseIntegration
from knowledge_base.conversation_manager import ConversationManager, ConversationState
from knowledge_base.llama_index.query_engine import QueryEngine

from langgraph_integration.nodes.state import AgentState, NodeType, ConversationStatus

logger = logging.getLogger(__name__)

class KBNode:
    """
    Knowledge Base node for LangGraph.
    
    This node processes queries and generates responses.
    """
    
    def __init__(
        self,
        kb_integration: Optional[KnowledgeBaseIntegration] = None,
        query_engine: Optional[QueryEngine] = None,
        conversation_manager: Optional[ConversationManager] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        include_sources: bool = True,
        timeout: int = 55  # Default timeout of 55 seconds
    ):
        """
        Initialize the KB node.
        
        Args:
            kb_integration: Existing KB integration to use
            query_engine: Query engine to use if creating new integration
            conversation_manager: Conversation manager if creating new integration
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response generation
            include_sources: Whether to include sources in the response
            timeout: Timeout in seconds for KB queries
        """
        self.include_sources = include_sources
        self.timeout = timeout
        
        if kb_integration:
            self.kb = kb_integration
        elif query_engine and conversation_manager:
            self.kb = KnowledgeBaseIntegration(
                query_engine=query_engine,
                conversation_manager=conversation_manager,
                temperature=temperature,
                max_tokens=max_tokens,
                use_cache=True  # Enable caching for better latency
            )
        else:
            raise ValueError("Either kb_integration or both query_engine and conversation_manager must be provided")
    
    async def process(self, state: AgentState) -> AsyncIterator[AgentState]:
        """
        Process the input state and generate a response with robust error handling.
        
        Args:
            state: The current agent state
            
        Yields:
            Updated agent state with response
        """
        # Update state
        state.current_node = NodeType.KB
        state.status = ConversationStatus.THINKING
        
        # Start timing
        start_time = time.time()
        
        try:
            # Check for query
            if not state.query:
                if state.transcription:
                    state.query = state.transcription
                    logger.info(f"Using transcription as query: '{state.query}'")
                else:
                    logger.error("No query provided to KB node")
                    state.error = "No query provided to KB node"
                    state.status = ConversationStatus.ERROR
                    yield state
                    return
            
            # Query the knowledge base with timeout protection
            logger.info(f"Querying knowledge base with: '{state.query}'")
            
            try:
                # Set a timeout for the entire knowledge base query operation
                kb_task = asyncio.create_task(self.kb.query(state.query, include_context=self.include_sources))
                result = await asyncio.wait_for(kb_task, timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning("Knowledge base query timed out, using simplified response")
                # Create a simplified response when timeout occurs
                result = {
                    "response": "I'm processing your question, but it's taking longer than expected. Could you please rephrase your question more specifically?",
                    "query": state.query,
                    "total_time": time.time() - start_time
                }
            
            # Debug: log the raw KB response
            logger.info(f"Knowledge base response: {result}")
            
            # Check for errors
            if "error" in result and result["error"]:
                state.error = result["error"]
                # Don't set ERROR status - instead provide a graceful response
                state.response = "I'm sorry, I couldn't find specific information about that. Could you try asking in a different way?"
            else:
                # Update state with results
                state.response = result.get("response", "")
            
            # Ensure we have a response even if empty
            if not state.response:
                logger.warning("Empty response generated by knowledge base")
                state.response = "I don't have enough information to answer that question specifically. Could you ask something about our pricing plans or features?"
            
            # Add context and sources if available
            if self.include_sources:
                if "context" in result:
                    state.context = result.get("context", "")
                if "sources" in result:
                    state.sources = result["sources"]
            
            # Update status - always proceed to TTS even if there were processing issues
            state.status = ConversationStatus.RESPONDING
            state.next_node = NodeType.TTS
            
            # Save timing information
            state.timings["kb"] = time.time() - start_time
            if "retrieval_time" in result:
                state.timings["retrieval_time"] = result.get("retrieval_time", 0.0)
            if "llm_time" in result:
                state.timings["llm_time"] = result.get("llm_time", 0.0)
            if "cache_hit" in result and result["cache_hit"]:
                state.timings["cache_hit"] = True
            
            # Add to history
            state.history.append({
                "role": "assistant",
                "content": state.response
            })
            
            # Debug log the state after processing
            logger.info(f"KB processing complete. Response length: {len(state.response)} chars")
            
        except Exception as e:
            logger.error(f"Error in KB node: {e}")
            state.error = f"KB error: {str(e)}"
            # Provide a fallback response rather than error state
            state.response = "I apologize, but I'm having trouble accessing that information right now. Is there something else I can help with?"
            state.status = ConversationStatus.RESPONDING
            state.next_node = NodeType.TTS
            
            # Record the timing
            state.timings["kb"] = time.time() - start_time
        
        # Return updated state
        yield state
    
    async def cleanup(self):
        """Clean up resources."""
        logger.debug("Cleaning up KB node")
        # Reset conversation state
        if hasattr(self, 'kb') and self.kb:
            self.kb.reset_conversation()