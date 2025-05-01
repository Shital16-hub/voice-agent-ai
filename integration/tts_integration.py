"""
TTS Integration module for Voice AI Agent using Google Cloud TTS.

This module provides classes and functions for integrating text-to-speech
capabilities with the Voice AI Agent system.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

from text_to_speech import GoogleCloudTTS, RealTimeResponseHandler, AudioProcessor

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration for Voice AI Agent using Google Cloud TTS.
    
    Provides an abstraction layer for TTS functionality,
    handling initialization, single-text processing, and streaming capabilities.
    """
    
    def __init__(
        self,
        voice: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice: Voice name to use for Google Cloud TTS
            enable_caching: Whether to enable TTS caching
        """
        self.voice = voice
        self.enable_caching = enable_caching
        self.tts_client = None
        self.tts_handler = None
        self.initialized = False
        
        # Parameters for better conversation flow
        self.add_pause_after_speech = True
        self.pause_duration_ms = 250  # Reduced from 500ms to 250ms for faster pacing
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Initialize the Google Cloud TTS client with optimized settings
            self.tts_client = GoogleCloudTTS(
                voice_name=self.voice, 
                enable_caching=self.enable_caching
            )
            
            # Initialize the RealTimeResponseHandler for streaming
            self.tts_handler = RealTimeResponseHandler(tts_client=self.tts_client)
            
            self.initialized = True
            logger.info(f"Initialized TTS with voice: {self.voice or 'default'}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech with improved streaming response.
        
        This optimized version handles small text chunks more efficiently
        for better real-time responsiveness.
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Skip very short chunks that might cause stuttering
            if len(text.strip()) < 2:
                # For single words or very short chunks, we need to ensure we don't
                # create unnatural pauses. Check if it ends with punctuation
                if not any(text.strip().endswith(p) for p in ['.', '!', '?', ',', ';', ':']):
                    # For non-sentence-ending chunks, append a space to smooth synthesis
                    text = text.strip() + " "
            
            # First attempt: Try with plain text (no SSML)
            try:
                start_time = time.time()
                audio_data = await self.tts_client.synthesize(text, is_ssml=False)
                synthesis_time = time.time() - start_time
                logger.debug(f"TTS synthesis took {synthesis_time:.3f}s for {len(text)} chars")
            except Exception as first_error:
                logger.warning(f"Plain text TTS failed: {first_error}. Trying with default voice...")
                
                # Second attempt: Use a default/standard voice
                try:
                    # Create a temporary client with a Standard voice
                    temp_client = GoogleCloudTTS(voice_name="en-US-Standard-D")
                    audio_data = await temp_client.synthesize(text, is_ssml=False)
                except Exception as second_error:
                    logger.error(f"Both TTS attempts failed: {second_error}")
                    
                    # Last resort: Generate 300ms of silence or a simple tone
                    silence_size = int(16000 * 0.3 * 2)  # 300ms of silence at 16kHz, 16-bit
                    audio_data = b'\x00' * silence_size
            
            # Ensure even number of bytes & add pause
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            if self.add_pause_after_speech:
                silence_size = int(16000 * (self.pause_duration_ms / 1000) * 2)
                silence_data = b'\x00' * silence_size
                audio_data = audio_data + silence_data
            
            return audio_data
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            # Return silence as last resort
            silence_size = int(16000 * 0.3 * 2)
            return b'\x00' * silence_size
    
    async def process_text_streaming(
        self,
        text: str,
        callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process text with streaming for immediate response.
        
        Args:
            text: Text to convert to speech
            callback: Callback to handle audio chunks
            
        Returns:
            Statistics about the processing
        """
        if not self.initialized:
            await self.init()
        
        # Split text into natural chunks for streaming
        chunks = self._split_into_natural_chunks(text)
        
        # Start measuring time
        start_time = time.time()
        total_chunks = 0
        total_audio_bytes = 0
        
        try:
            # Process each chunk with minimal buffering
            for chunk in chunks:
                audio_data = await self.text_to_speech(chunk)
                await callback(audio_data)
                
                # Update stats
                total_chunks += 1
                total_audio_bytes += len(audio_data)
                
                # Small delay between chunks to sound more natural
                # Only add delay between chunks, not at the end
                if total_chunks < len(chunks):
                    await asyncio.sleep(0.01)  # 10ms delay between chunks
                    
            # Calculate total time
            total_time = time.time() - start_time
            
            return {
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "processing_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in streaming TTS: {e}")
            return {
                "error": str(e),
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes
            }
    
    def _split_into_natural_chunks(self, text: str) -> List[str]:
        """
        Split text into natural chunks for better streaming.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Start with sentence splitting (periods, exclamations, question marks)
        sentence_delimiters = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        chunks = []
        
        # Try sentence splitting first
        remaining_text = text
        for delimiter in sentence_delimiters:
            if delimiter in remaining_text:
                parts = remaining_text.split(delimiter)
                for i in range(len(parts) - 1):
                    chunks.append(parts[i] + delimiter[0])
                remaining_text = parts[-1]
        
        # Add any remaining text
        if remaining_text.strip():
            chunks.append(remaining_text.strip())
        
        # If we have very long sentences, break them by punctuation
        phrase_delimiters = [', ', '; ', '- ', ':\n', ';\n', ',\n']
        result_chunks = []
        
        for chunk in chunks:
            if len(chunk) > 100:  # Long chunks
                remaining = chunk
                for delimiter in phrase_delimiters:
                    if delimiter in remaining:
                        parts = remaining.split(delimiter)
                        for i in range(len(parts) - 1):
                            result_chunks.append(parts[i] + delimiter[0])
                        remaining = parts[-1]
                # Add remaining text
                if remaining.strip():
                    result_chunks.append(remaining.strip())
            else:
                result_chunks.append(chunk)
        
        # If chunks are still too long or no punctuation, use word chunking
        final_chunks = []
        
        for chunk in result_chunks:
            if len(chunk) > 100:  # Still long chunks
                words = chunk.split()
                current_chunk = ""
                
                for word in words:
                    if len(current_chunk) + len(word) + 1 > 50:  # ~50 chars per chunk
                        if current_chunk:
                            final_chunks.append(current_chunk)
                        current_chunk = word
                    else:
                        if current_chunk:
                            current_chunk += " " + word
                        else:
                            current_chunk = word
                
                if current_chunk:
                    final_chunks.append(current_chunk)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
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
            # Yield silent audio as fallback
            silence_size = int(16000 * 0.3 * 2)  # 300ms of silence
            yield b'\x00' * silence_size
    
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
            self.tts_handler = RealTimeResponseHandler(tts_client=self.tts_client)
        
        # Process each text chunk
        total_chunks = 0
        total_audio_bytes = 0
        
        try:
            # Buffer for accumulating small text chunks
            buffer = ""
            
            async for chunk in text_chunks:
                if not chunk or not chunk.strip():
                    continue
                
                # Add to buffer
                buffer += chunk
                
                # Process the buffer if it contains sentence-ending punctuation or is large enough
                if any(p in buffer for p in ['.', '!', '?']) or len(buffer) >= 50:
                    # Process the accumulated text with simpler SSML for speed
                    try:
                        simple_ssml = "<speak>" + buffer + "</speak>"
                        audio_data = await self.tts_client.synthesize(simple_ssml, is_ssml=True)
                    except Exception as ssml_error:
                        # Fall back to plain text
                        logger.warning(f"SSML synthesis failed: {ssml_error}. Using plain text.")
                        audio_data = await self.tts_client.synthesize(buffer, is_ssml=False)
                    
                    # Track statistics
                    total_chunks += 1
                    total_audio_bytes += len(audio_data)
                    
                    # Send audio to callback
                    await audio_callback(audio_data)
                    
                    # Clear buffer
                    buffer = ""
            
            # Process any remaining text in buffer
            if buffer:
                audio_data = await self.tts_client.synthesize(buffer, is_ssml=False)
                total_chunks += 1
                total_audio_bytes += len(audio_data)
                await audio_callback(audio_data)
        
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
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.tts_handler:
            try:
                await self.tts_handler.stop()
            except Exception as e:
                logger.error(f"Error during TTS cleanup: {e}")