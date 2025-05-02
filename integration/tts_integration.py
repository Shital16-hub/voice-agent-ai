"""
TTS Integration module for Voice AI Agent.

This module provides functions for integrating text-to-speech
capabilities with the Voice AI Agent system.
"""
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

from text_to_speech import DeepgramTTS, RealTimeResponseHandler, AudioProcessor

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration for Voice AI Agent.
    
    Provides an abstraction layer for TTS functionality, handling initialization,
    single-text processing, and streaming capabilities.
    """
    
    def __init__(
        self,
        voice: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice: Voice ID to use for Deepgram TTS
            enable_caching: Whether to enable TTS caching
        """
        self.voice = voice
        self.enable_caching = enable_caching
        self.tts_client = None
        self.tts_handler = None
        self.initialized = False
        
        # Parameters for better conversation flow
        self.add_pause_after_speech = True
        self.pause_duration_ms = 500  # 500ms pause after speech
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Initialize the DeepgramTTS client with linear16 format
            self.tts_client = DeepgramTTS(
                voice=self.voice, 
                enable_caching=self.enable_caching,
                container_format="linear16",  # Use linear16 for PCM WAV
                sample_rate=16000  # Set sample rate for telephony
            )
            
            # Initialize the RealTimeResponseHandler
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
            
            self.initialized = True
            logger.info(f"Initialized TTS with voice: {self.voice or 'default'}, format: linear16")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Get audio data from TTS client
            audio_data = await self.tts_client.synthesize(text)
            
            # Ensure the audio data has an even number of bytes
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
                logger.debug("Padded audio data to make even length")
            
            # Add a short pause after speech for better conversation flow
            if self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(16000 * (self.pause_duration_ms / 1000) * 2)  # 16-bit samples
                silence_data = b'\x00' * silence_size
                
                # Append silence to audio data
                audio_data = audio_data + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after speech")
            
            return audio_data
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            raise
    
    async def text_to_speech_streaming(
        self, 
        text_generator: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech conversion.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Yields:
            Audio data chunks
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Track if we need to add the final pause
            needs_final_pause = False
            
            async for audio_chunk in self.tts_client.synthesize_streaming(text_generator):
                # Ensure each chunk has an even number of bytes
                if len(audio_chunk) % 2 != 0:
                    audio_chunk = audio_chunk + b'\x00'
                
                # Only the last chunk should get the pause
                needs_final_pause = True
                yield audio_chunk
            
            # Add a pause at the end of the complete audio stream
            if needs_final_pause and self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(16000 * (self.pause_duration_ms / 1000) * 2)  # 16-bit samples
                silence_data = b'\x00' * silence_size
                yield silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause at end of streaming audio")
                
        except Exception as e:
            logger.error(f"Error in streaming text to speech: {e}")
            raise
    
    async def process_realtime_text(
        self,
        text_chunks: AsyncIterator[str],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process text chunks in real-time and generate speech.
        
        Args:
            text_chunks: Async iterator of text chunks
            audio_callback: Callback to handle audio data
            
        Returns:
            Statistics about the processing
        """
        if not self.initialized:
            await self.init()
        
        # Start measuring time
        start_time = time.time()
        
        # Reset the TTS handler for this new session
        if self.tts_handler:
            await self.tts_handler.stop()
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
        
        # Process each text chunk
        total_chunks = 0
        total_audio_bytes = 0
        
        try:
            async for chunk in text_chunks:
                if not chunk or not chunk.strip():
                    continue
                
                # Process the text chunk
                audio_data = await self.text_to_speech(chunk)
                
                # Track statistics
                total_chunks += 1
                total_audio_bytes += len(audio_data)
                
                # Send audio to callback
                await audio_callback(audio_data)
                
                # Log progress periodically
                if total_chunks % 10 == 0:
                    logger.debug(f"Processed {total_chunks} text chunks")
        
        except Exception as e:
            logger.error(f"Error processing realtime text: {e}")
            return {
                "error": str(e),
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "elapsed_time": time.time() - start_time
            }
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        
        return {
            "total_chunks": total_chunks,
            "total_audio_bytes": total_audio_bytes,
            "elapsed_time": elapsed_time,
            "avg_chunk_size": total_audio_bytes / total_chunks if total_chunks > 0 else 0
        }
    
    async def process_ssml(self, ssml: str) -> bytes:
        """
        Process SSML text and convert to speech.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            audio_data = await self.tts_client.synthesize_with_ssml(ssml)
            # Ensure even number of bytes
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            # Add a pause at the end if needed
            if self.add_pause_after_speech:
                silence_size = int(16000 * (self.pause_duration_ms / 1000) * 2)
                silence_data = b'\x00' * silence_size
                audio_data = audio_data + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after SSML speech")
                
            return audio_data
        except Exception as e:
            logger.error(f"Error in SSML processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.tts_handler:
            try:
                await self.tts_handler.stop()
            except Exception as e:
                logger.error(f"Error during TTS cleanup: {e}")