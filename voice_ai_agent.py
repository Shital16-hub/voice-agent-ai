"""
Voice AI Agent main class that coordinates all components with improved noise handling.
Generic version that works with any knowledge base.
"""
import os
import logging
import asyncio
from typing import Optional, Dict, Any
import numpy as np
from scipy import signal

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, PARAMETER_PRESETS
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import DeepgramTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class that coordinates all components with noise handling improvements."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        whisper_model_path: str = 'models/base.en',
        llm_temperature: float = 0.7,
        # Add noise handling parameters with generic context
        whisper_initial_prompt: str = "This is a telephone conversation with a customer. The customer may ask questions about products, services, pricing, or features. Transcribe exactly what is spoken, filtering out noise, static, and line interference.",
        whisper_temperature: float = 0.0,
        whisper_no_context: bool = True,
        whisper_preset: Optional[str] = "default"
    ):
        """Initialize the Voice AI Agent with improved noise handling configuration."""
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.whisper_model_path = whisper_model_path
        self.llm_temperature = llm_temperature
        
        # Store noise handling parameters
        self.whisper_initial_prompt = whisper_initial_prompt
        self.whisper_temperature = whisper_temperature
        self.whisper_no_context = whisper_no_context
        self.whisper_preset = whisper_preset
        
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Add noise floor tracking for adaptive threshold
        self.noise_floor = 0.005
        self.noise_samples = []
        self.max_noise_samples = 20
        
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio for improved speech recognition.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        try:
            # Update noise floor from quiet sections
            self._update_noise_floor(audio)
            
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(6, 100/(16000/2), 'highpass')
            audio = signal.filtfilt(b, a, audio)
            
            # Apply band-pass filter for telephony frequency range
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            audio = signal.filtfilt(b, a, audio)
            
            # Apply pre-emphasis to boost high frequencies
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
            # Apply noise gate with adaptive threshold
            threshold = self.noise_floor * 3.0
            audio = np.where(np.abs(audio) < threshold, 0, audio)
            
            # Normalize audio level
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * (0.9 / max_val)
                
            return audio
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio  # Return original if processing fails
    
    def _update_noise_floor(self, audio: np.ndarray) -> None:
        """Update noise floor estimate from quiet sections."""
        # Find quiet sections (bottom 10% of energy)
        frame_size = min(len(audio), int(0.02 * 16000))  # 20ms frames
        if frame_size <= 1:
            return
            
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        frame_energies = [np.mean(np.square(frame)) for frame in frames]
        
        if len(frame_energies) > 0:
            # Sort energies and take bottom 10%
            sorted_energies = sorted(frame_energies)
            quiet_count = max(1, len(sorted_energies) // 10)
            quiet_energies = sorted_energies[:quiet_count]
            
            # Update noise samples
            self.noise_samples.extend(quiet_energies)
            
            # Limit sample count
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples = self.noise_samples[-self.max_noise_samples:]
            
            # Update noise floor with safety limits
            if self.noise_samples:
                self.noise_floor = max(
                    0.001,  # Minimum
                    min(0.02, np.percentile(self.noise_samples, 90) * 1.5)  # Maximum
                )
        
    async def init(self):
        """Initialize all components with noise handling optimizations."""
        logger.info("Initializing Voice AI Agent components with noise optimization...")
        
        # Initialize speech recognizer with improved noise handling parameters
        self.speech_recognizer = StreamingWhisperASR(
            model_path=self.whisper_model_path,
            language="en",
            n_threads=4,
            chunk_size_ms=2000,
            vad_enabled=True,
            single_segment=True,
            # Use our noise handling parameters
            temperature=self.whisper_temperature,
            initial_prompt=self.whisper_initial_prompt,
            no_context=self.whisper_no_context,
            # Use the preset if provided
            preset=self.whisper_preset
        )
        
        # Set parameters directly on the model if possible
        if hasattr(self.speech_recognizer, 'model'):
            model = self.speech_recognizer.model
            
            # Set parameters if supported
            if hasattr(model, 'temperature'):
                model.temperature = self.whisper_temperature
                
            if hasattr(model, 'initial_prompt'):
                model.initial_prompt = self.whisper_initial_prompt
                
            if hasattr(model, 'no_context'):
                model.no_context = self.whisper_no_context
                
            if hasattr(model, 'single_segment'):
                model.single_segment = True
                
            if hasattr(model, 'beam_size'):
                model.beam_size = 5  # Wider beam search
        
        # Initialize STT integration with enhanced audio processing
        self.stt_integration = EnhancedSTTIntegration(
            speech_recognizer=self.speech_recognizer,
            language="en",
            agent=self  # Pass self for access to audio processing
        )
        
        # Initialize document store and index manager
        doc_store = DocumentStore()
        index_manager = IndexManager(storage_dir=self.storage_dir)
        await index_manager.init()
        
        # Initialize query engine
        self.query_engine = QueryEngine(
            index_manager=index_manager, 
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature
        )
        await self.query_engine.init()
        
        # Initialize conversation manager with optimized parameters
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            # Skip greeting for better telephone experience
            skip_greeting=True
        )
        await self.conversation_manager.init()
        
        # Initialize TTS client
        self.tts_client = DeepgramTTS()
        
        logger.info("Voice AI Agent initialization complete with noise handling optimizations")
        
    async def process_audio(self, audio_data, callback=None):
        """
        Process audio data with noise handling optimizations.
        
        Args:
            audio_data: Audio data as numpy array
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Process audio for better speech recognition
        enhanced_audio = self._process_audio(audio_data)
        
        # Use STT integration for processing with enhanced audio
        result = await self.stt_integration.transcribe_audio_data(enhanced_audio, callback=callback)
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(transcription)
            
            return {
                "transcription": transcription,
                "response": response.get("response", ""),
                "status": "success"
            }
        else:
            logger.info("Invalid or empty transcription")
            return {
                "status": "invalid_transcription",
                "transcription": result.get("transcription", ""),
                "error": "No valid speech detected"
            }
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return (self.speech_recognizer is not None and 
                self.conversation_manager is not None and 
                self.query_engine is not None)
                
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        # Any cleanup needed here
        
        # Example: If there's an active conversation
        if self.conversation_manager:
            self.conversation_manager.reset()


class EnhancedSTTIntegration(STTIntegration):
    """Enhanced STT integration with better audio processing for telephony."""
    
    def __init__(self, speech_recognizer, language="en", agent=None):
        """Initialize with reference to agent for audio processing."""
        super().__init__(speech_recognizer, language)
        self.agent = agent
        
    async def transcribe_audio_data(self, audio_data, is_short_audio=False, callback=None):
        """Override to enhance audio before transcription."""
        # Process audio for better speech recognition if agent available
        if self.agent and hasattr(self.agent, '_process_audio'):
            # Process if it's a numpy array
            if isinstance(audio_data, np.ndarray):
                audio_data = self.agent._process_audio(audio_data)
        
        # Call parent method with enhanced audio
        return await super().transcribe_audio_data(audio_data, is_short_audio, callback)
        
    def cleanup_transcription(self, text):
        """Enhanced cleanup with more patterns for noise."""
        if not text:
            return ""
            
        # Add these patterns to your existing patterns
        additional_patterns = [
            r'\(.*?noise.*?\)', r'\[.*?noise.*?\]',
            r'\(.*?background.*?\)', r'\[.*?music.*?\]',
            r'\(.*?static.*?\)', r'\[.*?unclear.*?\]',
            r'\(.*?inaudible.*?\)', r'\<.*?noise.*?\>',
            r'music playing', r'background noise',
            r'static'
        ]
        
        # Compile if needed and add to existing pattern
        if hasattr(self, 'non_speech_pattern'):
            # Add to existing pattern if possible
            pattern_string = self.non_speech_pattern.pattern
            for pattern in additional_patterns:
                if pattern not in pattern_string:
                    pattern_string += '|' + pattern
            import re
            self.non_speech_pattern = re.compile(pattern_string)
            
        # Rest of your existing code
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove common filler words at beginning of sentences
        import re
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Log changes if substantial
        if text != cleaned_text:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text