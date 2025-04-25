"""
Helper functions for the LangGraph integration.

This module provides utility functions for working with
the LangGraph-based Voice AI Agent.
"""
import os
import json
import time
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union

import numpy as np

from langgraph_integration.nodes.state import AgentState, ConversationStatus

logger = logging.getLogger(__name__)

def create_initial_state(
    audio_input: Optional[Union[bytes, np.ndarray]] = None,
    audio_file_path: Optional[str] = None,
    text_input: Optional[str] = None,
    conversation_id: Optional[str] = None,
    speech_output_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Create an initial state for the LangGraph.
    
    Args:
        audio_input: Audio input data
        audio_file_path: Path to audio file
        text_input: Direct text input
        conversation_id: Conversation ID (generated if None)
        speech_output_path: Path to save speech output
        metadata: Additional metadata
        
    Returns:
        Initial agent state
    """
    # Validate inputs
    if not any([audio_input, audio_file_path, text_input]):
        raise ValueError("At least one of audio_input, audio_file_path, or text_input must be provided")
    
    # Create state
    state = AgentState(
        audio_input=audio_input,
        audio_file_path=audio_file_path,
        text_input=text_input,
        conversation_id=conversation_id or str(uuid.uuid4()),
        speech_output_path=speech_output_path,
        metadata=metadata or {},
        status=ConversationStatus.IDLE,
        timings={"start_time": time.time()}
    )
    
    return state

async def save_state_history(
    state_history: List[AgentState],
    output_path: str
) -> None:
    """
    Save the state history to a file.
    
    Args:
        state_history: List of agent states
        output_path: Path to save the history
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert states to dictionaries
    state_dicts = []
    for state in state_history:
        # Skip binary data for readability
        state_dict = state.dict(exclude={"audio_input", "speech_output"})
        
        # Add metadata about binary fields
        if state.audio_input is not None:
            if isinstance(state.audio_input, np.ndarray):
                state_dict["audio_input_info"] = f"NumPy array with shape {state.audio_input.shape}"
            else:
                state_dict["audio_input_info"] = f"Binary data with size {len(state.audio_input)} bytes"
        
        if state.speech_output is not None:
            state_dict["speech_output_info"] = f"Binary data with size {len(state.speech_output)} bytes"
        
        state_dicts.append(state_dict)
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(state_dicts, f, indent=2, default=str)

def calculate_confidence(state: AgentState) -> float:
    """
    Calculate overall confidence score for the agent's response.
    
    Args:
        state: Current agent state
        
    Returns:
        Confidence score between 0 and 1
    """
    # Start with transcription confidence if available
    confidence = state.transcription_confidence or 0.8
    
    # Adjust based on other factors
    
    # 1. Short responses might indicate uncertainty
    if state.response and len(state.response.split()) < 5:
        confidence *= 0.8
    
    # 2. No context retrieval might indicate lack of knowledge
    if not state.context:
        confidence *= 0.9
    
    # 3. Error indicators in response
    uncertainty_phrases = ["i'm not sure", "i don't know", "i'm uncertain", "cannot", "unable to"]
    if state.response and any(phrase in state.response.lower() for phrase in uncertainty_phrases):
        confidence *= 0.7
    
    return min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1

def should_handoff_to_human(state: AgentState, threshold: float = 0.7) -> bool:
    """
    Determine if the conversation should be handed off to a human.
    
    Args:
        state: Current agent state
        threshold: Confidence threshold for handoff
        
    Returns:
        Whether to hand off to a human
    """
    # Always hand off if explicitly required
    if state.requires_human:
        return True
    
    # Hand off on errors
    if state.error:
        return True
    
    # Hand off on low confidence
    confidence = calculate_confidence(state)
    if confidence < threshold:
        return True
    
    # Check for explicit requests for human in transcription
    human_request_phrases = ["speak to a human", "talk to a person", "speak to a person", "human operator"]
    if state.transcription and any(phrase in state.transcription.lower() for phrase in human_request_phrases):
        return True
    
    return False

class StateTracker:
    """
    Utility class for tracking state changes during graph execution.
    Enhanced with latency profiling and performance metrics.
    """
    
    def __init__(self, save_path: Optional[str] = None, latency_targets: Optional[Dict[str, float]] = None):
        """
        Initialize the state tracker.
        
        Args:
            save_path: Path to save state history (if None, history is kept in memory only)
            latency_targets: Target latency for each stage (in seconds)
        """
        self.history = []
        self.save_path = save_path
        self.latency_targets = latency_targets or {
            "stt": 0.5,
            "kb": 0.7, 
            "tts": 0.5,
            "total": 2.0
        }
        
        # Add performance metrics tracking
        self.performance_metrics = {
            "stt_latency": [],
            "kb_latency": [],
            "tts_latency": [],
            "total_latency": [],
            "target_met_count": 0,
            "total_requests": 0
        }
    
    def add_state(self, state) -> None:
        """
        Add a state to the history with enhanced performance tracking.
        
        Args:
            state: Agent state to add (could be any state object type)
        """
        try:
            # Create a dictionary representation of the state
            if hasattr(state, 'dict') and callable(getattr(state, 'dict')):
                # For Pydantic models
                state_dict = state.dict()
            elif hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
                # For objects with to_dict method
                state_dict = state.to_dict()
            elif hasattr(state, 'model_dump') and callable(getattr(state, 'model_dump')):
                # For newer Pydantic models
                state_dict = state.model_dump()
            elif hasattr(state, '__dict__'):
                # For objects with __dict__
                state_dict = state.__dict__.copy()
            else:
                # For dict-like objects
                try:
                    state_dict = dict(state)
                except:
                    # Last resort, create a minimal dictionary
                    state_dict = {"timestamp": time.time()}
            
            # Extract essential fields
            self._extract_essential_fields(state, state_dict)
            
            # Extract latency metrics for performance tracking
            self._extract_latency_metrics(state, state_dict)
            
            # Add to history
            self.history.append(state_dict)
            
            # Debug log
            self._log_state_addition(state)
                
        except Exception as e:
            logger.warning(f"Failed to add state to history: {e}")
            # Add minimal state info as fallback
            try:
                minimal_state = {"timestamp": time.time()}
                self._extract_essential_fields(state, minimal_state)
                self.history.append(minimal_state)
            except Exception:
                logger.warning("Failed to add even minimal state info")
    
    def _extract_essential_fields(self, state, state_dict: Dict[str, Any]) -> None:
        """
        Extract essential fields from the state object to the state dictionary.
        
        Args:
            state: Original state object
            state_dict: State dictionary to update
        """
        # Key fields to extract for debugging and analysis
        essential_fields = [
            "transcription", "response", "error", "current_node", "next_node", 
            "status", "conversation_id"
        ]
        
        for field in essential_fields:
            try:
                if hasattr(state, field):
                    value = getattr(state, field)
                    if field not in state_dict or state_dict[field] is None:
                        state_dict[f"{field}_debug"] = str(value)
            except Exception:
                pass
        
        # Extract timing information
        try:
            if hasattr(state, "timings"):
                timings = getattr(state, "timings")
                if isinstance(timings, dict):
                    # If timings is already in the state_dict, merge them
                    if "timings" not in state_dict:
                        state_dict["timings"] = {}
                    
                    for key, value in timings.items():
                        state_dict["timings"][key] = value
                        
                    # Ensure start_time exists
                    if "start_time" not in state_dict["timings"]:
                        state_dict["timings"]["start_time"] = time.time()
        except Exception:
            # Ensure timings exists with at least start_time
            if "timings" not in state_dict:
                state_dict["timings"] = {"start_time": time.time()}
    
    def _extract_latency_metrics(self, state, state_dict: Dict[str, Any]) -> None:
        """
        Extract latency metrics from state for performance tracking.
        
        Args:
            state: Original state object
            state_dict: State dictionary
        """
        # Check for timings data
        if "timings" not in state_dict:
            return
            
        timings = state_dict["timings"]
        
        # Record individual stage latencies
        for stage in ["stt", "kb", "tts"]:
            if stage in timings and isinstance(timings[stage], (int, float)):
                self.record_latency(stage, timings[stage])
                
        # Calculate and record total latency if possible
        if "start_time" in timings and isinstance(timings["start_time"], (int, float)):
            total_time = time.time() - timings["start_time"]
            self.record_latency("total", total_time)
            
            # Track whether this request met the target latency
            if total_time <= self.latency_targets.get("total", 2.0):
                self.performance_metrics["target_met_count"] += 1
            
            self.performance_metrics["total_requests"] += 1
    
    def _log_state_addition(self, state) -> None:
        """
        Log information about the added state.
        
        Args:
            state: State object that was added
        """
        try:
            current_node = getattr(state, 'current_node', None)
            next_node = getattr(state, 'next_node', None)
            logger.debug(f"Added state to history: {current_node} -> {next_node}")
        except Exception:
            logger.debug("Added state to history (no node info available)")
    
    # Add latency tracking methods
    def record_latency(self, stage: str, latency: float):
        """
        Record latency for a specific stage.
        
        Args:
            stage: Stage name ("stt", "kb", "tts", "total")
            latency: Latency value in seconds
        """
        if f"{stage}_latency" in self.performance_metrics:
            self.performance_metrics[f"{stage}_latency"].append(latency)
        
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics based on recorded latencies.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Calculate average latencies
        for stage in ["stt", "kb", "tts", "total"]:
            latencies = self.performance_metrics.get(f"{stage}_latency", [])
            if latencies:
                metrics[f"avg_{stage}_latency"] = sum(latencies) / len(latencies)
                metrics[f"max_{stage}_latency"] = max(latencies)
                metrics[f"min_{stage}_latency"] = min(latencies)
                
                # Calculate percentage of requests meeting target
                target = self.latency_targets.get(stage, 0)
                if target > 0:
                    met_target = sum(1 for l in latencies if l <= target)
                    metrics[f"{stage}_target_met_pct"] = met_target / len(latencies) * 100
        
        # Calculate overall target metrics
        total_requests = self.performance_metrics["total_requests"]
        if total_requests > 0:
            metrics["overall_target_met_pct"] = (self.performance_metrics["target_met_count"] / total_requests) * 100
        
        return metrics
    
    async def save_history(self) -> None:
        """Save the state history to a file if save_path is set."""
        if not self.save_path:
            return
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
            
            # Process binary data for serialization
            processed_history = []
            for state in self.history:
                processed_state = {}
                for key, value in state.items():
                    if key in ['audio_input', 'speech_output']:
                        if value is not None:
                            # For binary data, just store metadata
                            try:
                                processed_state[key + '_info'] = f"Binary data of length {len(value) if hasattr(value, '__len__') else 'unknown'}"
                            except:
                                processed_state[key + '_info'] = "Binary data (details unavailable)"
                    else:
                        # For other fields, store the value
                        processed_state[key] = value
                processed_history.append(processed_state)
            
            # Add performance metrics to the saved history
            performance_metrics = self.calculate_performance_metrics()
            processed_history.append({
                "type": "performance_metrics",
                "metrics": performance_metrics,
                "timestamp": time.time()
            })
            
            # Write to file
            with open(self.save_path, "w") as f:
                json.dump(processed_history, f, indent=2, default=str)
                
            logger.info(f"Saved state history to {self.save_path}")
        except Exception as e:
            logger.warning(f"Error saving state history: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation with performance metrics.
        
        Returns:
            Summary dictionary
        """
        if not self.history:
            return {"error": "No history available"}
        
        try:
            first_state = self.history[0]
            last_state = self.history[-1]
            
            # Extract specific information for summary
            transcriptions = []
            responses = []
            
            # Look for transcriptions and responses in various fields
            for state in self.history:
                for key in state:
                    if 'transcription' in key.lower() and state[key]:
                        if state[key] not in transcriptions and state[key] != 'None':
                            transcriptions.append(state[key])
                    
                    if 'response' in key.lower() and state[key]:
                        if state[key] not in responses and state[key] != 'None':
                            responses.append(state[key])
            
            # Get timing information
            start_time = None
            if isinstance(first_state.get("timings"), dict) and "start_time" in first_state["timings"]:
                start_time = first_state["timings"]["start_time"]
            elif "timestamp" in first_state:
                start_time = first_state["timestamp"]
            
            # Calculate duration
            duration = 0
            if start_time:
                duration = time.time() - start_time
            
            # Add performance metrics
            performance_metrics = self.calculate_performance_metrics()
            
            # Build summary
            summary = {
                "conversation_id": first_state.get("conversation_id", "unknown"),
                "start_time": start_time,
                "end_time": time.time(),
                "duration": duration,
                "status": last_state.get("status", "unknown"),
                "num_turns": len([s for s in self.history if s.get("current_node") == "STT"]),
                "transcriptions": transcriptions,
                "responses": responses,
                "state_count": len(self.history),
                "performance_metrics": performance_metrics
            }
            
            # Check if latency target was met
            if duration > 0:
                summary["target_latency"] = self.latency_targets.get("total", 2.0)
                summary["met_target"] = duration <= summary["target_latency"]
                summary["percentage_of_target"] = (duration / summary["target_latency"]) * 100
            
            return summary
        
        except Exception as e:
            logger.warning(f"Error generating summary: {e}")
            return {
                "error": f"Error generating summary: {str(e)}",
                "num_states": len(self.history)
            }
    
    def _calculate_duration(self, first_state: Dict[str, Any]) -> float:
        """
        Calculate the conversation duration.
        
        Args:
            first_state: First state in the history
            
        Returns:
            Duration in seconds
        """
        try:
            start_time = None
            
            # Try to get start time from timings
            if isinstance(first_state.get("timings"), dict):
                start_time = first_state["timings"].get("start_time")
            
            # If no start time found, use timestamp or current time
            if start_time is None:
                start_time = first_state.get("timestamp", time.time())
            
            return time.time() - start_time
        except Exception:
            return 0.0