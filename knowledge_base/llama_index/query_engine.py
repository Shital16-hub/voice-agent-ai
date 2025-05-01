async def query_with_streaming(
    self,
    query_text: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Query the knowledge base with improved streaming response generation.
    
    Optimized for real-time word-by-word output to TTS.
    
    Args:
        query_text: Query text
        chat_history: Optional chat history
        
    Yields:
        Response chunks
    """
    if not self.is_initialized:
        await self.init()

    try:
        # Retrieve relevant context - this happens first
        retrieval_start = time.time()
        retrieval_results = await self.retrieve_with_sources(query_text)
        results = retrieval_results.get("results", [])
        context = self.format_retrieved_context(results)
        retrieval_time = time.time() - retrieval_start
        
        # Log retrieval info to help diagnose any issues
        logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")

        # Format system prompt with context
        system_prompt = format_system_prompt(
            base_prompt="You are an AI assistant that answers questions based on the provided information. "
                        "If the information doesn't contain the answer, acknowledge this clearly. "
                        "Keep your answers concise and direct.",
            retrieved_context=context
        )

        # Create messages
        messages = create_chat_messages(
            system_prompt=system_prompt,
            user_message=query_text,
            chat_history=chat_history
        )

        full_response = ""

        try:
            # Improve streaming for real-time TTS
            # Stream response in smallest possible chunks with token-by-token processing
            streaming_response = await self.llm.astream_chat(messages)
            
            # Prepare to collect full sentences for better TTS
            sentence_buffer = ""
            
            async for chunk in streaming_response:
                # Get just the text delta/chunk
                chunk_text = chunk.delta if hasattr(chunk, 'delta') else chunk.content
                
                # Skip empty chunks
                if not chunk_text:
                    continue
                    
                # Add to full response
                full_response += chunk_text
                
                # Add to sentence buffer
                sentence_buffer += chunk_text
                
                # Check if we have a complete sentence, phrase, or enough words 
                # to make a good TTS chunk
                complete_sentence = False
                
                # Check for sentence-ending punctuation
                if any(p in sentence_buffer[-1:] for p in ['.', '!', '?']):
                    complete_sentence = True
                # Or phrase-ending punctuation
                elif any(p in sentence_buffer[-1:] for p in [',', ':', ';']) and len(sentence_buffer.split()) > 3:
                    complete_sentence = True
                # Or we have enough words for a natural phrase
                elif len(sentence_buffer.split()) >= 6:
                    complete_sentence = True
                
                # Send complete sentences or phrases for better TTS quality
                if complete_sentence:
                    yield {
                        "chunk": sentence_buffer,
                        "done": False,
                        "sources": retrieval_results.get("sources", [])
                    }
                    # Reset buffer
                    sentence_buffer = ""
                # For single token outputs, also output small chunks periodically
                # to maintain responsiveness
                elif len(sentence_buffer) >= 20:
                    yield {
                        "chunk": sentence_buffer,
                        "done": False,
                        "sources": retrieval_results.get("sources", [])
                    }
                    # Reset buffer
                    sentence_buffer = ""

            # Send any remaining text in the buffer
            if sentence_buffer:
                yield {
                    "chunk": sentence_buffer,
                    "done": False,
                    "sources": retrieval_results.get("sources", [])
                }

            # Final completion signal with full response for reference
            yield {
                "chunk": "",
                "full_response": full_response,
                "done": True,
                "sources": retrieval_results.get("sources", [])
            }

        except Exception as stream_error:
            logger.error(f"Error streaming response: {stream_error}")
            yield {
                "chunk": "\nError streaming response.",
                "done": True,
                "error": str(stream_error)
            }
    
    except Exception as e:
        logger.error(f"Error in query_with_streaming: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        yield {
            "chunk": "Error processing your query.",
            "done": True,
            "error": str(e)
        }