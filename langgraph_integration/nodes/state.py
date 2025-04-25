"""
State definitions for the LangGraph-based Voice AI Agent.

This module defines the state schema and transitions for the LangGraph
orchestration of the Voice AI Agent.
"""
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

class NodeType(Enum):
    """Types of nodes in the LangGraph."""
    STT = auto()
    KB = auto()
    TTS = auto()
    ROUTER = auto()
    HUMAN_HANDOFF = auto()

class ConversationStatus(Enum):
    """Status of the conversation."""
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    RESPONDING = auto()
    COMPLETED = auto()
    ERROR = auto()
    HUMAN_HANDOFF = auto()

class AgentState(BaseModel):
    """
    State for the LangGraph agent.
    
    This defines the schema for the state that flows between nodes
    in the LangGraph.
    """
    # Input state
    audio_input: Optional[Union[bytes, np.ndarray]] = Field(
        default=None, 
        description="Audio input from the user"
    )
    audio_file_path: Optional[str] = Field(
        default=None,
        description="Path to an audio file to process"
    )
    text_input: Optional[str] = Field(
        default=None,
        description="Direct text input from the user"
    )
    
    # STT state
    transcription: Optional[str] = Field(
        default=None,
        description="Transcribed text from audio input"
    )
    transcription_confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the transcription"
    )
    interim_transcriptions: List[str] = Field(
        default_factory=list,
        description="Interim transcriptions during processing"
    )
    
    # KB state
    query: Optional[str] = Field(
        default=None,
        description="Query for the knowledge base"
    )
    response: Optional[str] = Field(
        default=None,
        description="Response from the knowledge base"
    )
    context: Optional[str] = Field(
        default=None,
        description="Context retrieved from the knowledge base"
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources for the knowledge base response"
    )
    
    # TTS state
    speech_output: Optional[bytes] = Field(
        default=None,
        description="Speech output data"
    )
    speech_output_path: Optional[str] = Field(
        default=None,
        description="Path to save speech output"
    )
    
    # Conversation state
    conversation_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the conversation"
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation history"
    )
    status: ConversationStatus = Field(
        default=ConversationStatus.IDLE,
        description="Current status of the conversation"
    )
    
    # System state
    current_node: Optional[NodeType] = Field(
        default=None,
        description="Current node being processed"
    )
    next_node: Optional[NodeType] = Field(
        default=None,
        description="Next node to process"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any"
    )
    requires_human: bool = Field(
        default=False,
        description="Whether human intervention is required"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    # Timing information
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information for performance analysis"
    )
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True