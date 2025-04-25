"""
LangGraph-based Voice AI Agent.

This module provides the main implementation of the Voice AI Agent
using LangGraph for orchestration, enabling more flexible and
powerful conversation flows.
"""
import os
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union, Tuple, AsyncIterator

import numpy as np
from langgraph.graph import StateGraph
# Update END to whatever is used in your version of LangGraph
try:
    from langgraph.graph import END
except ImportError:
    # If END is not defined in your version of LangGraph, use a string identifier
    END = "end"

from voice_ai_agent import VoiceAIAgent
from integration.tts_integration import TTSIntegration
from integration.kb_integration import KnowledgeBaseIntegration
from integration.stt_integration import STTIntegration

from langgraph_integration.nodes import (
    STTNode, 
    KBNode, 
    TTSNode, 
    AgentState, 
    NodeType, 
    ConversationStatus
)
from langgraph_integration.utils.helpers import (
    create_initial_state,
    save_state_history,
    should_handoff_to_human,
    StateTracker
)
from langgraph_integration.config import LangGraphConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class VoiceAILangGraph:
    """
    LangGraph-based Voice AI Agent.
    
    This class provides a LangGraph implementation of the Voice AI Agent,
    enabling more flexible and powerful conversation flows.
    """
    
    def __init__(
        self,
        voice_ai_agent: Optional[VoiceAIAgent] = None,
        stt_integration: Optional[STTIntegration] = None,
        kb_integration: Optional[KnowledgeBaseIntegration] = None,
        tts_integration: Optional[TTSIntegration] = None,
        config: Optional[LangGraphConfig] = None
    ):
        """
        Initialize the LangGraph-based Voice AI Agent.
        
        Args:
            voice_ai_agent: Existing VoiceAIAgent to use for components
            stt_integration: STT integration
            kb_integration: KB integration
            tts_integration: TTS integration
            config: Configuration for the LangGraph
        """
        self.voice_ai_agent = voice_ai_agent
        self.stt_integration = stt_integration
        self.kb_integration = kb_integration
        self.tts_integration = tts_integration
        self.config = config or DEFAULT_CONFIG
        
        # Nodes
        self.stt_node = None
        self.kb_node = None
        self.tts_node = None
        
        # Graph
        self.graph = None
        self.compiled_graph = None
        
        # State tracking
        self.state_tracker = StateTracker(
            self.config.state_history_path if self.config.save_state_history else None,
            latency_targets={
                "stt": self.config.target_stt_latency,
                "kb": self.config.target_kb_latency,
                "tts": self.config.target_tts_latency,
                "total": self.config.target_total_latency
            }
        )
        
        # Telephony callbacks
        self.audio_callback = None
    
    async def init(self) -> None:
        """Initialize the LangGraph and all components."""
        logger.info("Initializing VoiceAILangGraph")
        
        # Initialize the base agent if provided and needed
        if self.voice_ai_agent and (not self.voice_ai_agent.speech_recognizer or
                                  not self.voice_ai_agent.query_engine or
                                  not self.voice_ai_agent.conversation_manager):
            await self.voice_ai_agent.init()
        
        # Initialize nodes
        await self._init_nodes()
        
        # Create the graph
        self._create_graph()
        
        logger.info("VoiceAILangGraph initialization complete")
    
    async def _init_nodes(self) -> None:
        """Initialize all nodes for the graph."""
        # STT Node
        if self.stt_integration:
            self.stt_node = STTNode(stt_integration=self.stt_integration)
        elif self.voice_ai_agent and self.voice_ai_agent.speech_recognizer:
            self.stt_node = STTNode(speech_recognizer=self.voice_ai_agent.speech_recognizer)
        else:
            self.stt_node = STTNode(model_path=self.config.stt_model, language=self.config.stt_language)
            await self.stt_node.stt.init(model_path=self.config.stt_model)
        
        # KB Node
        if self.kb_integration:
            self.kb_node = KBNode(
                kb_integration=self.kb_integration,
                timeout=self.config.kb_timeout,
                include_sources=self.config.kb_include_sources
            )
        elif self.voice_ai_agent and self.voice_ai_agent.query_engine and self.voice_ai_agent.conversation_manager:
            self.kb_node = KBNode(
                query_engine=self.voice_ai_agent.query_engine,
                conversation_manager=self.voice_ai_agent.conversation_manager,
                temperature=self.config.kb_temperature,
                max_tokens=self.config.kb_max_tokens,
                include_sources=self.config.kb_include_sources,
                timeout=self.config.kb_timeout
            )
        else:
            raise ValueError("Either kb_integration or voice_ai_agent with query_engine and conversation_manager must be provided")
        
        # TTS Node with optional callback
        if self.tts_integration:
            self.tts_node = TTSNode(
                tts_integration=self.tts_integration,
                output_callback=self.audio_callback
            )
        else:
            self.tts_node = TTSNode(
                voice=self.config.tts_voice,
                output_callback=self.audio_callback
            )
            await self.tts_node.tts.init()
    
    def _create_graph(self) -> None:
        """Create the LangGraph with optimized execution paths."""
        # Create state graph
        self.graph = StateGraph(AgentState)
        
        # Add nodes
        self.graph.add_node("stt", self.stt_node.process)
        self.graph.add_node("kb", self.kb_node.process)
        self.graph.add_node("tts", self.tts_node.process)
        
        # Add fast path for optimized processing (bypassing slower nodes when possible)
        # For example, if text is provided directly, bypass STT
        if self.config.enable_fast_paths:
            self.graph.add_conditional_edges(
                "stt",
                self._route_from_stt,
                {
                    "kb": "kb", 
                    "tts": "tts",  # Direct path for previously cached responses
                    "error": END
                }
            )
        else:
            # Standard routing
            self.graph.add_conditional_edges(
                "stt",
                self._route_from_stt,
                {
                    "kb": "kb",
                    "error": END
                }
            )
        
        # KB routing
        self.graph.add_conditional_edges(
            "kb",
            self._route_from_kb,
            {
                "tts": "tts",
                "error": END
            }
        )
        
        # TTS routing
        self.graph.add_conditional_edges(
            "tts",
            self._route_from_tts,
            {
                "end": END,
                "error": END
            }
        )
        
        # Set entry point
        self.graph.set_entry_point("stt")
        
        # Compile the graph
        self.compiled_graph = self.graph.compile()
    
    def _route_from_stt(self, state) -> str:
        """
        Route from STT node based on state.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name
        """
        # Safe access to state properties
        has_error = getattr(state, "error", None)
        status = getattr(state, "status", None)
        is_error_status = status == ConversationStatus.ERROR if status else False
        
        # Fast path: If we have cached response and direct text input, go straight to TTS
        if (hasattr(state, "metadata") and getattr(state, "metadata", {}).get("cache_hit") 
            and getattr(state, "response", None)):
            return "tts"
        
        if has_error or is_error_status:
            return "error"
        return "kb"
    
    def _route_from_kb(self, state) -> str:
        """
        Route from KB node based on state.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name
        """
        # Safe access to state properties
        has_error = getattr(state, "error", None)
        status = getattr(state, "status", None)
        is_error_status = status == ConversationStatus.ERROR if status else False
        
        if has_error or is_error_status:
            return "error"
        return "tts"
    
    def _route_from_tts(self, state) -> str:
        """
        Route from TTS node based on state.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name
        """
        # Safe access to state properties
        has_error = getattr(state, "error", None)
        status = getattr(state, "status", None)
        is_error_status = status == ConversationStatus.ERROR if status else False
        
        if has_error or is_error_status:
            return "error"
        return "end"
    
    def set_audio_callback(self, callback: Callable[[bytes], Awaitable[None]]) -> None:
        """
        Set a callback for audio output.
        
        This is useful for integration with telephony systems.
        
        Args:
            callback: Async function that receives audio data
        """
        self.audio_callback = callback
        
        # Update TTS node if already created
        if self.tts_node:
            self.tts_node.output_callback = callback
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline with enhanced latency tracking.
        
        Args:
            audio_file_path: Path to audio file
            speech_output_path: Path to save speech output
            metadata: Additional metadata
            
        Returns:
            Results dictionary
        """
        if not self.compiled_graph:
            await self.init()
        
        # Record start time for tracking
        start_time = time.time()
        
        # Add start time to metadata
        if metadata is None:
            metadata = {}
        metadata["start_time"] = start_time
        
        # Create initial state
        state = create_initial_state(
            audio_file_path=audio_file_path,
            speech_output_path=speech_output_path,
            metadata=metadata
        )
        
        # Make sure timings includes start_time
        if not hasattr(state, "timings") or not state.timings:
            state.timings = {"start_time": start_time}
        elif "start_time" not in state.timings:
            state.timings["start_time"] = start_time
        
        # Add initial state to state tracking
        try:
            self.state_tracker.add_state(state)
        except Exception as e:
            logger.warning(f"Failed to track initial state: {e}")
        
        # Run the graph
        try:
            print(f"Starting LangGraph pipeline processing for file: {audio_file_path}")
            
            # Record processing start time
            processing_start_time = time.time()
            
            # Execute the graph
            final_state = await self.compiled_graph.ainvoke(state)
            
            # Record processing end time
            processing_end_time = time.time()
            processing_duration = processing_end_time - processing_start_time
            
            # Add processing duration to timings
            try:
                if hasattr(final_state, "timings") and isinstance(final_state.timings, dict):
                    final_state.timings["total_processing_time"] = processing_duration
                elif hasattr(final_state, "timings"):
                    setattr(final_state, "timings", {"total_processing_time": processing_duration, "start_time": start_time})
            except Exception as e:
                logger.warning(f"Could not add processing time to state: {e}")
            
            # Track final state
            try:
                self.state_tracker.add_state(final_state)
            except Exception as e:
                logger.warning(f"Failed to track final state: {e}")
            
            # Save state history
            if self.config.save_state_history:
                try:
                    await self.state_tracker.save_history()
                except Exception as e:
                    logger.warning(f"Failed to save state history: {e}")
            
            # Record latencies for performance tracking
            total_time = time.time() - start_time
            
            # Record stage latencies
            if hasattr(final_state, "timings"):
                timings = final_state.timings
                if "stt" in timings:
                    self.state_tracker.record_latency("stt", timings["stt"])
                if "kb" in timings:
                    self.state_tracker.record_latency("kb", timings["kb"])
                if "tts" in timings:
                    self.state_tracker.record_latency("tts", timings["tts"])
                
                # Record total latency
                self.state_tracker.record_latency("total", total_time)
                
                # Check if met target
                self.state_tracker.performance_metrics["total_requests"] += 1
                if total_time <= self.config.target_total_latency:
                    self.state_tracker.performance_metrics["target_met_count"] += 1
            
            # Extract results from state
            results = self._extract_results_from_state(final_state)
            
            # Ensure certain fields are always present
            self._ensure_required_fields(results, final_state)
            
            # Ensure total time is accurate
            results["total_time"] = total_time
            
            # Add performance metrics
            results["performance"] = {
                "met_target": total_time <= self.config.target_total_latency,
                "target_latency": self.config.target_total_latency,
                "percentage_of_target": (total_time / self.config.target_total_latency) * 100
            }
            
            print(f"LangGraph pipeline completed in {results['total_time']:.2f} seconds " +
                f"({results['performance']['percentage_of_target']:.1f}% of {self.config.target_total_latency}s target)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running LangGraph: {e}", exc_info=True)
            
            # Create error results
            error_results = {
                "error": str(e),
                "status": "ERROR",
                "total_time": time.time() - start_time,
                "transcription": getattr(state, "transcription", ""),
                "response": getattr(state, "response", "")
            }
            
            return error_results
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data through the complete pipeline.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            speech_output_path: Path to save speech output
            metadata: Additional metadata
            
        Returns:
            Results dictionary
        """
        if not self.compiled_graph:
            await self.init()
        
        # Record start time for tracking
        start_time = time.time()
        
        # Add start time to metadata
        if metadata is None:
            metadata = {}
        metadata["start_time"] = start_time
        
        # Create initial state
        state = create_initial_state(
            audio_input=audio_data,
            speech_output_path=speech_output_path,
            metadata=metadata
        )
        
        # Make sure timings includes start_time
        if not hasattr(state, "timings") or not state.timings:
            state.timings = {"start_time": start_time}
        elif "start_time" not in state.timings:
            state.timings["start_time"] = start_time
            
        # Add initial state to state tracking
        try:
            self.state_tracker.add_state(state)
        except Exception as e:
            logger.warning(f"Failed to track initial state: {e}")
        
        # Run the graph
        try:
            # Record processing start time
            processing_start_time = time.time()
            
            # Execute the graph
            final_state = await self.compiled_graph.ainvoke(state)
            
            # Record processing end time
            processing_end_time = time.time()
            processing_duration = processing_end_time - processing_start_time
            
            # Add processing duration to timings if possible
            try:
                if hasattr(final_state, "timings") and isinstance(final_state.timings, dict):
                    final_state.timings["total_processing_time"] = processing_duration
                elif hasattr(final_state, "timings"):
                    # Try setting the attribute
                    setattr(final_state, "timings", {"total_processing_time": processing_duration, "start_time": start_time})
            except Exception as e:
                logger.warning(f"Could not add processing time to state: {e}")
            
            # Track final state
            try:
                self.state_tracker.add_state(final_state)
            except Exception as e:
                logger.warning(f"Failed to track final state: {e}")
            
            # Save state history
            if self.config.save_state_history:
                try:
                    await self.state_tracker.save_history()
                except Exception as e:
                    logger.warning(f"Failed to save state history: {e}")
            
            # Extract results from state
            results = self._extract_results_from_state(final_state)
            
            # Ensure certain fields are always present
            self._ensure_required_fields(results, final_state)
            
            # Ensure total time is accurate
            results["total_time"] = time.time() - start_time
            
            # Add performance metrics
            results["performance"] = {
                "met_target": results["total_time"] <= self.config.target_total_latency,
                "target_latency": self.config.target_total_latency,
                "percentage_of_target": (results["total_time"] / self.config.target_total_latency) * 100
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running LangGraph: {e}", exc_info=True)
            
            # Create error results
            error_results = {
                "error": str(e),
                "status": "ERROR",
                "total_time": time.time() - start_time,
                "transcription": getattr(state, "transcription", ""),
                "response": getattr(state, "response", "")
            }
            
            return error_results

    def _get_status_name(self, state) -> str:
        """
        Safely get the status name from a state object.
        
        Args:
            state: Agent state
            
        Returns:
            Status name as string
        """
        try:
            if hasattr(state, "status"):
                status = state.status
                if hasattr(status, "name"):
                    return status.name
                return str(status)
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"
    
    async def process_text(
        self,
        text: str,
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process text input through the pipeline (skipping STT).
        
        Args:
            text: Text input
            speech_output_path: Path to save speech output
            metadata: Additional metadata
            
        Returns:
            Results dictionary
        """
        if not self.compiled_graph:
            await self.init()
        
        # Record start time for tracking
        start_time = time.time()
        
        # Add start time to metadata
        if metadata is None:
            metadata = {}
        metadata["start_time"] = start_time
        
        # Create initial state
        state = create_initial_state(
            text_input=text,
            speech_output_path=speech_output_path,
            metadata=metadata
        )
        
        # Make sure timings includes start_time
        if not hasattr(state, "timings") or not state.timings:
            state.timings = {"start_time": start_time}
        elif "start_time" not in state.timings:
            state.timings["start_time"] = start_time
            
        # Add initial state to state tracking
        try:
            self.state_tracker.add_state(state)
        except Exception as e:
            logger.warning(f"Failed to track initial state: {e}")
        
        # Run the graph
        try:
            # Record processing start time
            processing_start_time = time.time()
            
            # Execute the graph with a timeout to ensure responsiveness
            try:
                # Use asyncio.wait_for to enforce a timeout
                execute_task = asyncio.create_task(self.compiled_graph.ainvoke(state))
                final_state = await asyncio.wait_for(execute_task, 
                                                     timeout=self.config.target_total_latency * 3)  # Triple the target latency
            except asyncio.TimeoutError:
                logger.warning(f"Graph execution timed out for text: {text}")
                # Create a fallback state with a reasonable response
                final_state = state
                final_state.response = "I'm having trouble processing your request in a timely manner. Could you try again with a simpler question?"
                final_state.status = ConversationStatus.ERROR
            
            # Record processing end time
            processing_end_time = time.time()
            processing_duration = processing_end_time - processing_start_time
            
            # Add processing duration to timings if possible
            try:
                if hasattr(final_state, "timings") and isinstance(final_state.timings, dict):
                    final_state.timings["total_processing_time"] = processing_duration
                elif hasattr(final_state, "timings"):
                    # Try setting the attribute
                    setattr(final_state, "timings", {"total_processing_time": processing_duration, "start_time": start_time})
            except Exception as e:
                logger.warning(f"Could not add processing time to state: {e}")
            
            # Track final state
            try:
                self.state_tracker.add_state(final_state)
            except Exception as e:
                logger.warning(f"Failed to track final state: {e}")
            
            # Save state history
            if self.config.save_state_history:
                try:
                    await self.state_tracker.save_history()
                except Exception as e:
                    logger.warning(f"Failed to save state history: {e}")
            
            # Extract results from state
            results = self._extract_results_from_state(final_state)
            
            # Ensure certain fields are always present
            self._ensure_required_fields(results, final_state)
            
            # Ensure total time is accurate
            results["total_time"] = time.time() - start_time
            
            # Add performance metrics
            results["performance"] = {
                "met_target": results["total_time"] <= self.config.target_total_latency,
                "target_latency": self.config.target_total_latency,
                "percentage_of_target": (results["total_time"] / self.config.target_total_latency) * 100
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running LangGraph: {e}", exc_info=True)
            
            # Create error results
            error_results = {
                "error": str(e),
                "status": "ERROR",
                "total_time": time.time() - start_time,
                "transcription": getattr(state, "transcription", ""),
                "response": getattr(state, "response", "")
            }
            
            return error_results
    
    async def process_streaming(
        self,
        stream_input: Union[AsyncIterator[np.ndarray], AsyncIterator[bytes], str],
        is_text: bool = False,
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process streaming input with streaming output.
        
        Args:
            stream_input: Audio stream or text
            is_text: Whether the input is text
            speech_output_path: Path to save final speech output
            metadata: Additional metadata
            
        Yields:
            Results for each processing step
        """
        # Not fully implemented yet - would need more complex streaming graph
        raise NotImplementedError("Streaming processing not implemented yet")
    
    def _extract_results_from_state(self, state) -> Dict[str, Any]:
        """
        Extract results from a state object, handling any type of state object.
        
        Args:
            state: State object from LangGraph
            
        Returns:
            Results dictionary
        """
        # Start time for total time calculation
        start_time = self._get_start_time(state)
        
        # Create a base results dictionary
        results = {
            "status": self._get_status_name(state),
            "total_time": time.time() - start_time
        }
        
        # Add timings if available
        try:
            if hasattr(state, "timings"):
                results["timings"] = getattr(state, "timings", {})
        except Exception:
            # Create minimal timings if none available
            results["timings"] = {"start_time": start_time}
        
        # Extract common fields in a safe way
        fields_to_extract = [
            "transcription", "response", "error", "conversation_id", 
            "sources", "history", "speech_output_path"
        ]
        
        for field in fields_to_extract:
            try:
                value = getattr(state, field, None)
                if value is not None:
                    # Special handling for binary data
                    if field == "speech_output" and value:
                        try:
                            results["speech_audio_size"] = len(value)
                        except:
                            results["speech_audio_size"] = "Unknown"
                    else:
                        results[field] = value
            except Exception:
                # Skip fields that can't be accessed
                pass
        
        # Add node information if available
        try:
            results["current_node"] = str(getattr(state, "current_node", "unknown"))
            results["next_node"] = str(getattr(state, "next_node", "unknown"))
        except Exception:
            pass
        
        # Calculate individual stage timings if available
        results["stage_timings"] = {}
        for stage in ["stt", "kb", "tts"]:
            try:
                if stage in getattr(state, "timings", {}):
                    results["stage_timings"][stage] = state.timings[stage]
            except Exception:
                pass
        
        # Calculate total time using stage timings if available
        if results["stage_timings"]:
            total_stage_time = sum(results["stage_timings"].values())
            if total_stage_time > 0:
                results["calculated_total_time"] = total_stage_time
        
        return results
    
    def _ensure_required_fields(self, results: Dict[str, Any], state) -> None:
        """
        Ensure required fields are present in the results dictionary.
        
        Args:
            results: Results dictionary to modify
            state: Original state object
        """
        # Always ensure these fields are present
        required_fields = ["transcription", "response"]
        
        for field in required_fields:
            if field not in results:
                # Try to get from state, or use empty string as fallback
                try:
                    results[field] = getattr(state, field, "")
                except:
                    results[field] = ""
    
    def _get_start_time(self, state) -> float:
        """
        Safely get the start time from a state object.
        
        Args:
            state: State object
            
        Returns:
            Start time as float
        """
        # Try multiple ways to get the start time
        try:
            # Try getting from timings dictionary
            if hasattr(state, "timings") and isinstance(state.timings, dict) and "start_time" in state.timings:
                return state.timings["start_time"]
            
            # Try getting from metadata
            if hasattr(state, "metadata") and isinstance(state.metadata, dict) and "start_time" in state.metadata:
                return state.metadata["start_time"]
            
            # Try direct state attribute
            if hasattr(state, "start_time"):
                return state.start_time
            
            # If state is dict-like, try dictionary access
            try:
                if "timings" in state and isinstance(state["timings"], dict) and "start_time" in state["timings"]:
                    return state["timings"]["start_time"]
                if "start_time" in state:
                    return state["start_time"]
            except:
                pass
            
            # Last resort: check history
            if len(self.state_tracker.history) > 0:
                first_state = self.state_tracker.history[0]
                if "timings" in first_state and isinstance(first_state["timings"], dict) and "start_time" in first_state["timings"]:
                    return first_state["timings"]["start_time"]
            
            # If all else fails, return a time at least 1 second in the past
            return time.time() - 1.0
            
        except Exception as e:
            logger.warning(f"Error getting start time: {e}")
            return time.time() - 1.0  # Fallback to at least 1 second
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up VoiceAILangGraph")
        
        # Clean up nodes
        if hasattr(self.stt_node, 'cleanup'):
            await self.stt_node.cleanup()
        
        if hasattr(self.kb_node, 'cleanup'):
            await self.kb_node.cleanup()
        
        if hasattr(self.tts_node, 'cleanup'):
            await self.tts_node.cleanup()