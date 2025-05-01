#!/usr/bin/env python3
import re
import sys

# Path to the file
filepath = '/workspace/voice-ai/voice-agent-ai/telephony/websocket_handler.py'

with open(filepath, 'r') as file:
    content = file.read()

# Find the query_with_streaming section in _process_audio method
pattern = r'async for chunk in self\.pipeline\.query_engine\.query_with_streaming\(transcription\):.*?# Reset agent speaking state'
match = re.search(pattern, content, re.DOTALL)

if match:
    old_code = match.group(0)
    
    # Create the new code with accumulated response
    new_code = '''
                    # Accumulate the full response first
                    logger.info(f"Getting full response for: {transcription}")
                    
                    # Collect the full response
                    full_response = ""
                    async for chunk in self.pipeline.query_engine.query_with_streaming(transcription):
                        # Check if cancelled due to barge-in
                        if self.speech_cancellation_event.is_set():
                            logger.info("Response generation cancelled due to barge-in")
                            break
                        
                        chunk_text = chunk.get("chunk", "")
                        if not chunk_text:
                            # Check for final response
                            if chunk.get("done", False) and "full_response" in chunk:
                                full_response = chunk.get("full_response", "")
                                logger.info(f"Got final response: {full_response}")
                            continue
                        
                        # Add to the full response
                        full_response += chunk_text
                    
                    # Now that we have the complete response, send it to TTS
                    if full_response and not self.speech_cancellation_event.is_set():
                        logger.info(f"Converting complete response to speech: {full_response}")
                        
                        # Generate speech for the complete response
                        speech_audio = await self.pipeline.tts_integration.text_to_speech(full_response)
                        
                        # Convert to mulaw for Twilio
                        mulaw_audio = self.audio_processor.pcm_to_mulaw(speech_audio)
                        
                        # Send back to Twilio
                        logger.info(f"Sending complete audio response ({len(mulaw_audio)} bytes)")
                        await self._send_audio(mulaw_audio, ws)
                    
                    # Reset agent speaking state'''
                    
    # Replace the code
    new_content = content.replace(old_code, new_code)
    
    # Write the updated content back to the file
    with open(filepath, 'w') as file:
        file.write(new_content)
    
    print("Successfully updated the WebSocketHandler._process_audio method to accumulate full responses")
else:
    print("Could not find the pattern to replace. Manual editing required.")
    sys.exit(1)
