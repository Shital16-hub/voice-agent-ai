"""
Text processing utilities for knowledge base.
"""
import re
from typing import List, Dict, Any, Optional, Set

def preprocess_text(text: str) -> str:
    """
    Preprocess text for embedding.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = text.strip()
    
    return text

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        
    Returns:
        List of keywords
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Split into words
    words = processed_text.split()
    
    # Filter out short words and duplicates
    keywords = list(set([word for word in words if len(word) >= min_length]))
    
    return keywords

def extract_entities(text: str) -> List[str]:
    """
    Extract named entities from text.
    
    Args:
        text: Input text
        
    Returns:
        List of entities
    """
    try:
        import spacy
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            # If model not found, download it
            import subprocess
            subprocess.run([
                "python", "-m", "spacy", "download", "en_core_web_sm"
            ])
            nlp = spacy.load("en_core_web_sm")
        
        # Process text
        doc = nlp(text)
        
        # Extract entities
        entities = [ent.text for ent in doc.ents]
        
        return entities
    
    except ImportError:
        # Fallback to simple capitalized words
        words = text.split()
        entities = [word for word in words if word and word[0].isupper()]
        return entities

def count_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Simple approximation (English ~1.3 tokens per word)
    words = text.split()
    return int(len(words) * 1.3)

def truncate_text(text: str, max_tokens: int = 1000) -> str:
    """
    Truncate text to stay within token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum token count
        
    Returns:
        Truncated text
    """
    # Estimate current tokens
    current_tokens = count_tokens(text)
    
    if current_tokens <= max_tokens:
        return text
    
    # Calculate approximate ratio
    ratio = max_tokens / current_tokens
    
    # Calculate character cut
    char_limit = int(len(text) * ratio * 0.9)  # 10% safety margin
    
    # Truncate
    truncated = text[:char_limit]
    
    # Try to cut at sentence boundary
    last_period = truncated.rfind('.')
    if last_period > 0 and last_period > len(truncated) * 0.5:
        truncated = truncated[:last_period + 1]
    
    return truncated + " [truncated]"

def format_context_for_llm(contexts: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
    """
    Format context documents for LLM prompt.
    
    Args:
        contexts: List of context documents
        max_tokens: Maximum tokens for context
        
    Returns:
        Formatted context string
    """
    if not contexts:
        return "No relevant information found."
    
    formatted_parts = []
    
    # Format each context
    for i, context in enumerate(contexts):
        text = context.get("text", "")
        metadata = context.get("metadata", {})
        source = metadata.get("source", f"Source {i+1}")
        
        part = f"[Document {i+1}] Source: {source}\n{text}"
        formatted_parts.append(part)
    
    # Join all parts
    full_context = "\n\n".join(formatted_parts)
    
    # Truncate if too long
    truncated_context = truncate_text(full_context, max_tokens)
    
    return truncated_context