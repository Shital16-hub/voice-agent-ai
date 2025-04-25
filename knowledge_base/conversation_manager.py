"""
Conversation manager using state management with future LangGraph compatibility.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator, Union
from enum import Enum
import time
import json

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore

from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.llm_setup import get_ollama_llm, format_system_prompt, create_chat_messages

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """Enum for conversation states."""
    GREETING = "greeting"
    WAITING_FOR_QUERY = "waiting_for_query"
    RETRIEVING = "retrieving"
    GENERATING_RESPONSE = "generating_response"
    CLARIFYING = "clarifying"
    HUMAN_HANDOFF = "human_handoff"
    ENDED = "ended"

class ConversationTurn:
    """Represents a single turn in the conversation."""
    
    def __init__(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        state: ConversationState = ConversationState.WAITING_FOR_QUERY,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ConversationTurn.
        
        Args:
            query: User query
            response: System response
            retrieved_context: Retrieved documents
            state: Conversation state
            metadata: Additional metadata
        """
        self.query = query
        self.response = response
        self.retrieved_context = retrieved_context or []
        self.state = state
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "retrieved_context": self.retrieved_context,
            "state": self.state,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(
            query=data.get("query"),
            response=data.get("response"),
            retrieved_context=data.get("retrieved_context", []),
            state=ConversationState(data.get("state", ConversationState.WAITING_FOR_QUERY)),
            metadata=data.get("metadata", {})
        )

class ConversationManager:
    """
    Manage conversation state and flow with future LangGraph compatibility.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        session_id: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_temperature: float = 0.7,
        use_langgraph: bool = False,  # Flag for future LangGraph implementation
        skip_greeting: bool = False   # New parameter to skip greeting state
    ):
        """
        Initialize ConversationManager.
        
        Args:
            query_engine: QueryEngine instance
            session_id: Unique session identifier
            llm_model_name: Optional LLM model name
            llm_temperature: Temperature for sampling
            use_langgraph: Whether to use LangGraph (for future implementation)
            skip_greeting: Whether to skip the greeting state and start in WAITING_FOR_QUERY
        """
        self.query_engine = query_engine
        self.session_id = session_id or f"session_{int(time.time())}"
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.use_langgraph = use_langgraph
        self.skip_greeting = skip_greeting
        
        # LLM setup
        self.llm = None
        
        # Initialize conversation state - use skip_greeting to determine initial state
        self.current_state = ConversationState.WAITING_FOR_QUERY if skip_greeting else ConversationState.GREETING
        self.history: List[ConversationTurn] = []
        self.context_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # State for LangGraph (will be expanded later)
        self.graph_state = {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history": [],
            "context": None,
            "metadata": {}
        }
        
        logger.info(f"Initialized ConversationManager with session_id: {self.session_id}, initial_state: {self.current_state}")
    
    async def init(self):
        """Initialize dependencies."""
        # Initialize query engine if provided
        if self.query_engine:
            await self.query_engine.init()
        
        # Initialize LLM if not already set globally
        if not Settings.llm:
            self.llm = get_ollama_llm(
                model_name=self.llm_model_name,
                temperature=self.llm_temperature
            )
        else:
            self.llm = Settings.llm
        
        # Initialize LangGraph components if enabled (placeholder for future)
        if self.use_langgraph:
            logger.info("LangGraph integration will be implemented in a future update")
            # This will be expanded in the future LangGraph implementation
    
    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and move conversation forward.
        
        Args:
            user_input: User input text
            
        Returns:
            Response with next state and response text
        """
        # For future LangGraph implementation
        if self.use_langgraph:
            return await self._handle_user_input_langgraph(user_input)
        
        # Create new turn
        turn = ConversationTurn(
            query=user_input,
            state=self.current_state
        )
        
        # Check if this looks like a query even if we're in greeting state
        if self.current_state == ConversationState.GREETING and user_input and len(user_input.split()) > 2:
            logger.info("First message appears to be a query, handling as query instead of greeting")
            turn.state = ConversationState.WAITING_FOR_QUERY
            self.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Process based on current state
        if self.current_state == ConversationState.GREETING:
            # Handle greeting
            response = await self._handle_greeting(turn)
        elif self.current_state == ConversationState.WAITING_FOR_QUERY:
            # Handle query
            response = await self._handle_query(turn)
        elif self.current_state == ConversationState.CLARIFYING:
            # Handle clarification
            response = await self._handle_clarification(turn)
        elif self.current_state == ConversationState.HUMAN_HANDOFF:
            # Handle already in human handoff
            response = {
                "response": "I'll let the human agent know about your message.",
                "state": ConversationState.HUMAN_HANDOFF,
                "requires_human": True,
                "context": None
            }
        else:
            # Default handling
            response = await self._handle_query(turn)
        
        # Update turn with response and add to history
        turn.response = response["response"]
        turn.state = response["state"]
        self.history.append(turn)
        
        # Update current state
        self.current_state = response["state"]
        
        # Update graph state (for future LangGraph compatibility)
        self._update_graph_state(turn)
        
        return response
    
    async def _handle_user_input_langgraph(self, user_input: str) -> Dict[str, Any]:
        """
        LangGraph implementation of user input handling (placeholder for future).
        
        Args:
            user_input: User input text
            
        Returns:
            Response dictionary
        """
        # This will be implemented in the future LangGraph integration
        logger.info("LangGraph integration will be implemented in a future update")
        
        # For now, fall back to standard implementation
        return await self.handle_user_input(user_input)
    
    def _update_graph_state(self, turn: ConversationTurn):
        """
        Update the graph state with the latest turn information.
        Prepares for future LangGraph implementation.
        
        Args:
            turn: Latest conversation turn
        """
        # Update current state
        self.graph_state["current_state"] = turn.state
        
        # Add to history
        history_entry = {
            "role": "user",
            "content": turn.query
        }
        self.graph_state["history"].append(history_entry)
        
        if turn.response:
            response_entry = {
                "role": "assistant",
                "content": turn.response
            }
            self.graph_state["history"].append(response_entry)
        
        # Update context if available
        if turn.retrieved_context:
            self.graph_state["context"] = turn.retrieved_context
    
    async def _handle_greeting(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Handle greeting state.
        
        Args:
            turn: Current conversation turn
            
        Returns:
            Response dictionary
        """
        # Check if this is actually a query and not a greeting
        if turn.query and len(turn.query.split()) > 2:
            logger.info("Detected query in greeting state, handling as query")
            return await self._handle_query(turn)
            
        # Generate greeting response
        if self.llm:
            greeting_prompt = "Generate a friendly greeting for a customer service conversation."
            
            try:
                # Create system message
                system_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful AI assistant. Keep your response brief and friendly."
                )
                
                # Create user message
                user_message = ChatMessage(
                    role=MessageRole.USER,
                    content=greeting_prompt
                )
                
                # Generate response
                try:
                    response = await self.llm.achat([system_message, user_message])
                    response_text = response.message.content
                except AttributeError:
                    # Fallback to synchronous chat if async is not available
                    logger.info("Falling back to synchronous chat for greeting")
                    response = self.llm.chat([system_message, user_message])
                    response_text = response.message.content
            except Exception as e:
                logger.error(f"Error generating greeting: {e}")
                response_text = "Hello! How can I assist you today?"
        else:
            response_text = "Hello! How can I assist you today?"
        
        # Move to waiting for query
        return {
            "response": response_text,
            "state": ConversationState.WAITING_FOR_QUERY,
            "requires_human": False,
            "context": None
        }
    
    async def _handle_query(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Handle user query.
        
        Args:
            turn: Current conversation turn
            
        Returns:
            Response dictionary
        """
        query = turn.query
        
        # Check for human handoff request
        if self._check_for_human_handoff(query):
            return {
                "response": "I'll connect you with a human agent shortly. Please wait a moment.",
                "state": ConversationState.HUMAN_HANDOFF,
                "requires_human": True,
                "context": None
            }
        
        # Retrieve relevant documents
        context = None
        if self.query_engine:
            # Set state to retrieving
            turn.state = ConversationState.RETRIEVING
            
            try:
                # Get relevant documents
                retrieval_results = await self.query_engine.retrieve_with_sources(query)
                turn.retrieved_context = retrieval_results["results"]
                
                # Format context for LLM
                context = self.query_engine.format_retrieved_context(turn.retrieved_context)
                
                # Check if we have enough context
                if not turn.retrieved_context:
                    # No relevant information found
                    if self._should_clarify(query):
                        # Need clarification
                        return {
                            "response": self._generate_clarification_question(query),
                            "state": ConversationState.CLARIFYING,
                            "requires_human": False,
                            "context": None
                        }
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                context = None
        
        # Generate response
        turn.state = ConversationState.GENERATING_RESPONSE
        
        try:
            # Get conversation history for context
            conversation_history = self._format_conversation_history()
            
            # Create system prompt with context
            system_prompt = format_system_prompt(
                base_prompt="You are a helpful AI assistant. Answer the user's question based on the provided information.",
                retrieved_context=context
            )
            
            # Create messages
            messages = create_chat_messages(
                system_prompt=system_prompt,
                user_message=query,
                chat_history=conversation_history
            )
            
            # Generate response with fallback options
            try:
                # First try using the async chat method
                response = await self.llm.achat(messages)
                response_text = response.message.content
            except AttributeError:
                # If achat fails, try using the synchronous chat method
                logger.info("Falling back to synchronous chat method")
                response = self.llm.chat(messages)
                response_text = response.message.content
            
            # Log for debugging
            logger.info(f"LLM DIRECT RESPONSE: {response_text[:50]}...")
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            response_text = "I'm sorry, I'm having trouble processing your request right now."
        
        # Return response
        return {
            "response": response_text,
            "state": ConversationState.WAITING_FOR_QUERY,
            "requires_human": False,
            "context": context
        }
    
    async def _handle_clarification(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Handle clarification response from user.
        
        Args:
            turn: Current conversation turn
            
        Returns:
            Response dictionary
        """
        # Get original query from previous turn
        original_query = self.history[-1].query if self.history else ""
        
        # Combine original query with clarification
        combined_query = f"{original_query} {turn.query}"
        
        # Create new turn with combined query
        new_turn = ConversationTurn(
            query=combined_query,
            state=ConversationState.WAITING_FOR_QUERY
        )
        
        # Handle as normal query
        return await self._handle_query(new_turn)
    
    async def generate_streaming_response(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate a streaming response to user input.
        
        Args:
            user_input: User input text
            
        Returns:
            Async iterator of response chunks
        """
        # Check if this looks like a query even if we're in greeting state
        if self.current_state == ConversationState.GREETING and user_input and len(user_input.split()) > 2:
            logger.info("First message in streaming appears to be a query, handling as query instead of greeting")
            self.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Create new turn
        turn = ConversationTurn(
            query=user_input,
            state=self.current_state
        )
        
        # Check for human handoff
        if self._check_for_human_handoff(user_input):
            result = {
                "chunk": "I'll connect you with a human agent shortly. Please wait a moment.",
                "done": True,
                "requires_human": True,
                "state": ConversationState.HUMAN_HANDOFF
            }
            
            # Update turn and history
            turn.response = result["chunk"]
            turn.state = ConversationState.HUMAN_HANDOFF
            self.history.append(turn)
            self.current_state = ConversationState.HUMAN_HANDOFF
            
            # Update graph state
            self._update_graph_state(turn)
            
            yield result
            return
        
        # Retrieve relevant documents if appropriate
        context = None
        if self.query_engine:
            try:
                # Set state to retrieving
                turn.state = ConversationState.RETRIEVING
                
                # Get relevant documents
                retrieval_results = await self.query_engine.retrieve_with_sources(user_input)
                turn.retrieved_context = retrieval_results["results"]
                
                # Format context for LLM
                context = self.query_engine.format_retrieved_context(turn.retrieved_context)
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                context = None
        
        # Stream response
        turn.state = ConversationState.GENERATING_RESPONSE
        full_response = ""
        
        try:
            # Get conversation history
            conversation_history = self._format_conversation_history()
            
            # Create system prompt with context
            system_prompt = format_system_prompt(
                base_prompt="You are a helpful AI assistant. Answer the user's question based on the provided information.",
                retrieved_context=context
            )
            
            # Create messages
            messages = create_chat_messages(
                system_prompt=system_prompt,
                user_message=user_input,
                chat_history=conversation_history
            )
            
            try:
                # Try async streaming first
                stream_response = await self.llm.astream_chat(messages)
                
                # Now we can iterate through the streaming response
                async for chunk in stream_response:
                    # The attribute name might be delta or content depending on your LlamaIndex version
                    chunk_text = chunk.delta if hasattr(chunk, 'delta') else chunk.content
                    full_response += chunk_text
                    
                    yield {
                        "chunk": chunk_text,
                        "done": False,
                        "requires_human": False,
                        "state": ConversationState.GENERATING_RESPONSE
                    }
            except AttributeError:
                # If astream_chat is not available, fall back to non-streaming
                logger.info("Async streaming not available, falling back to regular chat")
                response = self.llm.chat(messages)
                response_text = response.message.content
                full_response = response_text
                
                # Send the full response as a single chunk
                yield {
                    "chunk": response_text,
                    "done": False,
                    "requires_human": False,
                    "state": ConversationState.GENERATING_RESPONSE
                }
            
            # Final result
            yield {
                "chunk": "",
                "full_response": full_response,
                "done": True,
                "requires_human": False,
                "state": ConversationState.WAITING_FOR_QUERY
            }
            
            # Update turn and history
            turn.response = full_response
            turn.state = ConversationState.WAITING_FOR_QUERY
            self.history.append(turn)
            self.current_state = ConversationState.WAITING_FOR_QUERY
            
            # Update graph state
            self._update_graph_state(turn)
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}", exc_info=True)
            
            error_message = "I'm sorry, I'm having trouble processing your request right now."
            yield {
                "chunk": error_message,
                "done": True,
                "requires_human": False,
                "state": ConversationState.WAITING_FOR_QUERY
            }
            
            # Update turn and history
            turn.response = error_message
            turn.state = ConversationState.WAITING_FOR_QUERY
            self.history.append(turn)
            self.current_state = ConversationState.WAITING_FOR_QUERY
            
            # Update graph state
            self._update_graph_state(turn)
    
    def _check_for_human_handoff(self, query: str) -> bool:
        """
        Check if user is requesting human handoff.
        
        Args:
            query: User query
            
        Returns:
            True if human handoff requested
        """
        # Simple keyword matching
        handoff_keywords = [
            "speak to a human",
            "talk to a person",
            "talk to someone",
            "speak to an agent",
            "connect me with",
            "real person",
            "human agent",
            "customer service",
            "representative"
        ]
        
        query_lower = query.lower()
        for keyword in handoff_keywords:
            if keyword in query_lower:
                return True
        
        return False
    
    def _should_clarify(self, query: str) -> bool:
        """
        Determine if we need clarification for the query.
        
        Args:
            query: User query
            
        Returns:
            True if clarification needed
        """
        # Check query length
        if len(query.split()) < 3:
            return True
        
        # Check for vagueness
        vague_terms = ["this", "that", "it", "thing", "stuff", "something"]
        query_lower = query.lower()
        for term in vague_terms:
            if term in query_lower.split():
                return True
        
        return False
    
    def _generate_clarification_question(self, query: str) -> str:
        """
        Generate a clarification question.
        
        Args:
            query: Original query
            
        Returns:
            Clarification question
        """
        # Simple template-based generation
        templates = [
            "Could you please provide more details about what you're looking for?",
            "I'd like to help, but I need a bit more information. Can you elaborate on your question?",
            "To better assist you, could you be more specific about what you need?",
            "I'm not sure I understand completely. Could you explain what you're looking for in more detail?",
            "Could you clarify what specifically you'd like to know about this topic?"
        ]
        
        import random
        return random.choice(templates)
    
    def _format_conversation_history(self) -> List[Dict[str, str]]:
        """
        Format conversation history for language model.
        
        Returns:
            Formatted conversation history
        """
        formatted_history = []
        
        # Add recent turns (up to last 5 turns)
        for turn in self.history[-5:]:
            if turn.query:
                formatted_history.append({
                    "role": "user",
                    "content": turn.query
                })
            
            if turn.response:
                formatted_history.append({
                    "role": "assistant",
                    "content": turn.response
                })
        
        return formatted_history
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation turns
        """
        return [turn.to_dict() for turn in self.history]
    
    def get_latest_context(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get most recently retrieved context.
        
        Returns:
            Retrieved context documents or None
        """
        # Find most recent turn with context
        for turn in reversed(self.history):
            if turn.retrieved_context:
                return turn.retrieved_context
        
        return None
    
    def reset(self):
        """Reset conversation state."""
        # Reset to initial state based on skip_greeting flag
        self.current_state = ConversationState.WAITING_FOR_QUERY if self.skip_greeting else ConversationState.GREETING
        self.history = []
        self.context_cache = {}
        
        # Reset graph state
        self.graph_state = {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history": [],
            "context": None,
            "metadata": {}
        }
        
        logger.info(f"Reset conversation for session: {self.session_id}")
    
    def get_state_for_transfer(self) -> Dict[str, Any]:
        """
        Get conversation state for human handoff.
        
        Returns:
            Dictionary with conversation state for transfer
        """
        # Create transfer state with relevant information
        transfer_state = {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history_summary": self._generate_history_summary(),
            "last_query": self.history[-1].query if self.history else None,
            "last_response": self.history[-1].response if self.history else None,
            "recent_context": self.get_latest_context()
        }
        
        return transfer_state
    
    def _generate_history_summary(self) -> str:
        """
        Generate a summary of conversation history.
        
        Returns:
            Summary text
        """
        if not self.history:
            return "No conversation history."
        
        # Count turns
        num_turns = len(self.history) // 2
        
        # Get key exchanges
        summary_parts = [f"Conversation with {num_turns} exchanges:"]
        
        for i, turn in enumerate(self.history):
            if turn.query:
                summary_parts.append(f"User: {turn.query}")
            if turn.response:
                # Truncate long responses
                response = turn.response
                if len(response) > 100:
                    response = response[:97] + "..."
                summary_parts.append(f"AI: {response}")
        
        return "\n".join(summary_parts)
    
    # LangGraph preparation - placeholder methods for future implementation
    def get_graph_state(self) -> Dict[str, Any]:
        """
        Get the current graph state.
        
        Returns:
            Current graph state
        """
        return self.graph_state
    
    def serialize_state(self) -> str:
        """
        Serialize the current state for LangGraph.
        
        Returns:
            Serialized state
        """
        return json.dumps(self.graph_state)
    
    @classmethod
    def deserialize_state(cls, serialized_state: str) -> Dict[str, Any]:
        """
        Deserialize a state for LangGraph.
        
        Args:
            serialized_state: Serialized state
            
        Returns:
            Deserialized state
        """
        return json.loads(serialized_state)