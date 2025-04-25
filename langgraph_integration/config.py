"""
Configuration for the LangGraph-based Voice AI Agent.

This module provides configuration settings and constants
for the LangGraph integration.
"""
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

@dataclass
class LangGraphConfig:
    """Configuration for the LangGraph integration."""
    
    # STT configuration
    stt_model: str = "tiny.en"
    stt_language: str = "en"
    
    # KB configuration
    kb_temperature: float = 0.7
    kb_max_tokens: int = 1024
    kb_include_sources: bool = True
    kb_timeout: int = 55     # Timeout in seconds for KB queries
    kb_use_cache: bool = True # Use response caching for improved latency
    
    # TTS configuration
    tts_voice: Optional[str] = None
    
    # Graph configuration
    enable_human_handoff: bool = True
    confidence_threshold: float = 0.7  # Threshold for human handoff
    enable_fast_paths: bool = True    # Enable optimized routing paths
    parallel_processing: bool = True  # Enable parallel processing where possible
    
    # Latency targets
    target_stt_latency: float = 0.5   # Target STT latency in seconds
    target_kb_latency: float = 0.7    # Target KB latency in seconds
    target_tts_latency: float = 0.5   # Target TTS latency in seconds
    target_total_latency: float = 2.0 # Target end-to-end latency in seconds
    
    # Debugging
    debug_mode: bool = False
    save_state_history: bool = False
    state_history_path: Optional[str] = None
    
    # Performance
    enable_streaming: bool = True
    
    # Custom node settings
    custom_node_settings: Dict[str, Any] = field(default_factory=dict)

# Default configuration optimized for performance
DEFAULT_CONFIG = LangGraphConfig(
    stt_model="tiny.en",
    kb_temperature=0.5,    # Lower temperature for faster responses
    kb_max_tokens=700,     # Lower max tokens for faster responses
    kb_timeout=55,         # 55 second timeout for KB queries 
    enable_fast_paths=True,
    parallel_processing=True,
    target_total_latency=2.0
)

# Router decision mapping
ROUTER_DECISIONS = {
    "stt_failure": "human_handoff",
    "kb_failure": "human_handoff",
    "tts_failure": "kb",  # Skip TTS on failure, return text response
    "low_confidence": "human_handoff",
    "default": "stt"
}

# Node mapping for state transitions
NODE_MAPPING = {
    "stt": "speech-to-text",
    "kb": "knowledge-base",
    "tts": "text-to-speech",
    "router": "router",
    "human_handoff": "human-handoff"
}