"""
Voice AI Agent main class that coordinates all components with Google Cloud Speech integration.
Generic version that works with any knowledge base.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable, List
import numpy as np
from scipy import signal

# Google Cloud Speech imports
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import DeepgramTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class that coordinates all components with Google Cloud Speech."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        credentials_path: Optional[str] = None,
        llm_temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with Google Cloud Speech.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            credentials_path: Path to Google Cloud credentials JSON file
            llm_temperature: LLM temperature for response generation
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.llm_temperature = llm_temperature
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        self.stt_keywords = kwargs.get('keywords', ['price', 'plan', 'cost', 'subscription', 'service', 'features', 'support'])
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Noise floor tracking for adaptive threshold
        self.noise_floor = 0.005
        self.noise_samples = []
        self.max_noise_samples = 20
        
        # Barge-in settings
        self.min_barge_in_energy = kwargs.get('min_barge_in_energy', 0.1)
        self.min_consecutive_chunks = kwargs.get('min_consecutive_chunks', 3)
        
        logger.info(f"Initialized VoiceAIAgent with Google Cloud Speech, model: {model_name}")
        
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio for better speech recognition.
        
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
        """
        Update noise floor estimate from quiet sections.
        
        Args:
            audio: Audio data as numpy array
        """
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
        """Initialize all components with Google Cloud Speech."""
        logger.info("Initializing Voice AI Agent components with Google Cloud Speech...")
        
        # Set up Google credentials
        if self.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
            logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to {self.credentials_path}")
        else:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not specified, using default credentials")
        
        # Initialize speech recognizer with Google Cloud Speech
        self.speech_recognizer = GoogleCloudStreamingSTT(
            credentials_path=self.credentials_path,
            language=self.stt_language,
            sample_rate=16000,
            interim_results=True,
            model="phone_call",
            enhanced=True
        )
        
        # Initialize STT integration 
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.stt_language
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
        
        logger.info("Voice AI Agent initialization complete with Google Cloud Speech")
        
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with Google Cloud Speech.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Process audio for better speech recognition
        if isinstance(audio_data, np.ndarray):
            audio_data = self._process_audio(audio_data)
        
        # Use STT integration for processing with Google Cloud Speech
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
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
    
    async def process_streaming_audio(
        self,
        audio_stream,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process streaming audio with real-time response.
        
        Args:
            audio_stream: Async iterator of audio chunks
            result_callback: Callback for streaming results
            
        Returns:
            Final processing stats
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Track stats
        start_time = time.time()
        chunks_processed = 0
        results_count = 0
        
        # Start streaming session
        await self.speech_recognizer.start_streaming()
        
        # Track barge-in state
        barge_in_detected = False
        consecutive_speech_chunks = 0
        
        try:
            # Process each audio chunk
            async for chunk in audio_stream:
                chunks_processed += 1
                
                # Process audio for better recognition if it's numpy array
                if isinstance(chunk, np.ndarray):
                    chunk = self._process_audio(chunk)
                
                # Check for potential barge-in based on energy
                if isinstance(chunk, np.ndarray):
                    energy = np.mean(np.abs(chunk))
                    if energy > self.min_barge_in_energy:
                        consecutive_speech_chunks += 1
                        if consecutive_speech_chunks >= self.min_consecutive_chunks:
                            barge_in_detected = True
                            logger.info("Barge-in detected through energy analysis")
                    else:
                        consecutive_speech_chunks = 0
                
                # Convert to bytes for Google Cloud Speech if needed
                if isinstance(chunk, np.ndarray):
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = chunk
                    
                # Process through Google Cloud Speech
                async def process_result(result):
                    # Handle barge-in detection from speech recognition
                    nonlocal barge_in_detected, consecutive_speech_chunks
                    
                    # Only handle final results
                    if result.is_final:
                        # Clean up transcription
                        transcription = self.stt_integration.cleanup_transcription(result.text)
                        
                        # Process if valid
                        if transcription and self.stt_integration.is_valid_transcription(transcription):
                            # Check if barge-in also detected through speech recognition
                            if self.speech_recognizer.is_barge_in_detected():
                                barge_in_detected = True
                                logger.info("Barge-in confirmed by speech recognition: " + transcription)
                            
                            # Get response from conversation manager
                            response = await self.conversation_manager.handle_user_input(transcription)
                            
                            # Format result
                            result_data = {
                                "transcription": transcription,
                                "response": response.get("response", ""),
                                "confidence": result.confidence,
                                "is_final": True,
                                "barge_in_detected": barge_in_detected
                            }
                            
                            nonlocal results_count
                            results_count += 1
                            
                            # Reset barge-in state after processing
                            barge_in_detected = False
                            consecutive_speech_chunks = 0
                            
                            # Call callback if provided
                            if result_callback:
                                await result_callback(result_data)
                
                # Process chunk
                await self.speech_recognizer.process_audio_chunk(audio_bytes, process_result)
                
            # Stop streaming session
            await self.speech_recognizer.stop_streaming()
            
            # Return stats
            return {
                "status": "complete",
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time,
                "barge_in_detected": barge_in_detected
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming audio: {e}")
            
            # Stop streaming session
            await self.speech_recognizer.stop_streaming()
            
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time,
                "barge_in_detected": barge_in_detected
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
        
        # Close Google Cloud streaming session if active
        if self.speech_recognizer:
            await self.speech_recognizer.stop_streaming()
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()
    
    async def test_speech_recognition(
        self, 
        audio_file_path: str
    ) -> Dict[str, Any]:
        """
        Test speech recognition with an audio file.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Recognition result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
            
        logger.info(f"Testing speech recognition with file: {audio_file_path}")
        
        # Use STT integration to transcribe the file
        result = await self.stt_integration.transcribe_audio_file(audio_file_path)
        
        return result
    
    async def handle_barge_in(self, force_reset: bool = False) -> None:
        """
        Handle barge-in by resetting the conversation flow.
        
        Args:
            force_reset: Whether to force a reset even if no barge-in detected
        """
        # Check if barge-in was detected
        if force_reset or self.speech_recognizer.is_barge_in_detected():
            logger.info("Handling barge-in: resetting conversation flow")
            
            # Reset barge-in detection
            self.speech_recognizer.reset_barge_in_detection()
            
            # Could implement additional actions here like:
            # - Stopping current TTS output
            # - Resetting conversation context
            # - Sending a notification to the telephony system
            
            return True
        
        return False