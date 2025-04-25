"""
LLM setup for LlamaIndex integration with Ollama.
"""
import logging
import os
from typing import Dict, Any, Optional, List

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-v0.2-q4_0")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # Increased to 60 seconds

# LLM parameters
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
DEFAULT_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "4096"))

# Add parallel processing flag
PARALLEL_PROCESSING = os.getenv("PARALLEL_PROCESSING", "True").lower() == "true"

def check_ollama_availability(base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Check if Ollama is running and accessible.
    
    Args:
        base_url: Ollama API base URL
        
    Returns:
        True if Ollama is available
    """
    import requests
    try:
        response = requests.get(f"{base_url}/api/tags")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama not available: {e}")
        return False

def get_ollama_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Ollama:
    """
    Get Ollama LLM for LlamaIndex integration.
    
    Args:
        model_name: Model name to use (defaults to environment or mistral:7b-instruct)
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters to pass to Ollama
        
    Returns:
        Configured Ollama LLM
    """
    # Get parameters, with fallbacks to defaults
    model = model_name or DEFAULT_MODEL
    base_url = kwargs.pop("base_url", OLLAMA_BASE_URL)
    request_timeout = kwargs.pop("request_timeout", OLLAMA_TIMEOUT)

    # Check if Ollama is available
    if not check_ollama_availability(base_url):
        raise ValueError(f"Ollama not available at {base_url}. Please ensure Ollama is running.")
    
    # Build parameters
    params = {
        "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        **kwargs
    }
    
    try:
        # Check if model exists
        import requests
        model_check = requests.post(f"{base_url}/api/show", json={"name": model})
        if model_check.status_code != 200:
            # Model doesn't exist, try to pull it
            logger.warning(f"Model {model} not found locally. Attempting to pull...")
            pull_response = requests.post(f"{base_url}/api/pull", json={"name": model})
            if pull_response.status_code != 200:
                raise ValueError(f"Failed to pull model {model}. Please pull it manually.")
            
        # Initialize Ollama LLM
        ollama_llm = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout,  # Pass the timeout explicitly
            **params
        )
        
        logger.info(f"Initialized Ollama LLM with model: {model}, timeout: {request_timeout}s")
        return ollama_llm
        
    except Exception as e:
        logger.error(f"Error initializing Ollama LLM: {e}")
        raise

def setup_global_llm(model_name: Optional[str] = None, **kwargs) -> Ollama:
    """
    Set up LlamaIndex to use Ollama LLM globally.
    
    Args:
        model_name: Model name to use
        **kwargs: Additional parameters to pass to Ollama
        
    Returns:
        The configured Ollama LLM instance
    """
    # Get Ollama LLM
    ollama_llm = get_ollama_llm(model_name, **kwargs)
    
    # Set as global LLM
    Settings.llm = ollama_llm
    
    logger.info(f"Set Ollama LLM as global LlamaIndex LLM with model: {ollama_llm.model}")
    return ollama_llm

def format_system_prompt(
    base_prompt: str,
    retrieved_context: Optional[str] = None
) -> str:
    """
    Format system prompt with optional retrieved context.
    
    Args:
        base_prompt: Base system prompt
        retrieved_context: Optional context from retrieval
        
    Returns:
        Formatted system prompt
    """
    if not retrieved_context:
        return base_prompt
    
    # Add the retrieved context to the system prompt
    return f"{base_prompt}\n\nRelevant information:\n{retrieved_context}"

def create_chat_messages(
    system_prompt: str,
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[ChatMessage]:
    """
    Create formatted chat messages for LLM.
    
    Args:
        system_prompt: System prompt
        user_message: User message
        chat_history: Optional chat history
        
    Returns:
        List of ChatMessage objects
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    ]
    
    # Add chat history if provided
    if chat_history:
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                messages.append(ChatMessage(role=MessageRole.USER, content=content))
            elif role == "assistant":
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
    
    # Add user message
    messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
    
    return messages

# LangGraph preparation - these functions will be enhanced later
def get_llm_node_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the input needed for the LLM node from the state.
    
    Args:
        state: The current state object
        
    Returns:
        The input for the LLM node
    """
    # For now, just return a simple extraction, will be enhanced for LangGraph
    return {
        "query": state.get("query", ""),
        "context": state.get("context", ""),
        "chat_history": state.get("history", [])
    }

def prepare_for_langgraph():
    """
    Placeholder function to prepare for LangGraph integration.
    Will be implemented when migrating to LangGraph.
    """
    logger.info("LangGraph integration will be implemented in a future update")
    pass