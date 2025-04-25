"""
Query engine for retrieving and generating information using LlamaIndex.
"""
import logging
import asyncio
import time  # Added missing import
from typing import Dict, List, Any, Optional, Tuple, AsyncIterator, Union

from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Settings

from knowledge_base.config import get_retriever_config
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.schema import Document
from knowledge_base.llama_index.llm_setup import get_ollama_llm, format_system_prompt, create_chat_messages

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Retrieve and generate information from the knowledge base using LlamaIndex.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        config: Optional[Dict[str, Any]] = None,
        llm_model_name: Optional[str] = None,
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 1024
    ):
        """
        Initialize QueryEngine.

        Args:
            index_manager: IndexManager instance
            config: Optional configuration dictionary
            llm_model_name: Optional LLM model name
            llm_temperature: Temperature for sampling
            llm_max_tokens: Maximum tokens to generate
        """
        self.index_manager = index_manager
        self.config = config or get_retriever_config()
        self.top_k = self.config["top_k"]
        self.min_score = self.config["min_score"]
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens

        self.retriever = None
        self.query_engine = None
        self.llm = None
        self.is_initialized = False

        logger.info(f"Initialized QueryEngine with top_k={self.top_k}, min_score={self.min_score}")

    async def init(self):
        """Initialize the query engine."""
        if self.is_initialized:
            return

        if not self.index_manager.is_initialized:
            await self.index_manager.init()

        self.llm = get_ollama_llm(
            model_name=self.llm_model_name,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens
        )

        Settings.llm = self.llm

        self.retriever = VectorIndexRetriever(
            index=self.index_manager.index,
            similarity_top_k=self.top_k,
            filters=None
        )

        from llama_index.core.response_synthesizers import ResponseMode
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=self.llm
        )

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            llm=self.llm
        )

        self.is_initialized = True
        logger.info("Query engine initialized with LlamaIndex LLM integration")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        """
        if not self.is_initialized:
            await self.init()

        top_k = top_k if top_k is not None else self.top_k

        try:
            query_bundle = QueryBundle(query_str=query)

            if top_k != self.retriever.similarity_top_k:
                self.retriever.similarity_top_k = top_k

            if filter_metadata:
                filters = {"metadata_filter": filter_metadata}
                self.retriever.filters = filters
            else:
                self.retriever.filters = None

            nodes = self.retriever.retrieve(query_bundle)

            documents = []
            for node in nodes:
                if min_score is not None and node.score < min_score:
                    continue

                doc = Document(
                    text=node.text,
                    metadata=node.metadata,
                    doc_id=node.id_
                )
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def retrieve_with_sources(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents with source information.
        """
        docs = await self.retrieve(query, top_k, min_score)

        if not docs:
            return {
                "query": query,
                "results": [],
                "sources": []
            }

        sources = []
        source_ids = set()
        results = []

        for doc in docs:
            metadata = doc.metadata
            result = {
                "id": doc.doc_id,
                "text": doc.text,
                "metadata": metadata,
                "score": metadata.get("score", 0.0)
            }
            results.append(result)

            source = metadata.get("source")
            if source and source not in source_ids:
                source_ids.add(source)
                source_info = {
                    "name": source,
                    "type": metadata.get("source_type", "unknown")
                }

                if metadata.get("file_path"):
                    source_info["file_path"] = metadata.get("file_path")
                    source_info["file_type"] = metadata.get("file_type")

                sources.append(source_info)

        return {
            "query": query,
            "results": results,
            "sources": sources
        }

    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string.
        """
        if not results:
            return "No relevant information found."

        context_parts = []

        for i, doc in enumerate(results):
            text = doc["text"]
            score = doc.get("score", 0)
            metadata = doc.get("metadata", {})
            source = metadata.get("source", f"Source {i+1}")

            context_parts.append(f"[Document {i+1}] Source: {source} (Relevance: {score:.2f})\n{text}")

        return "\n\n".join(context_parts)

    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        Query the knowledge base using LlamaIndex LLM.
        """
        if not self.is_initialized:
            await self.init()

        try:
            # Track timing
            start_time = time.time()

            query_bundle = QueryBundle(query_str=query_text)
            response = self.query_engine.query(query_bundle)

            source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
            sources = []
            for node in source_nodes:
                source = {
                    "text": node.text,
                    "score": node.score if hasattr(node, "score") else 0.0,
                    "metadata": node.metadata
                }
                sources.append(source)

            result = {
                "query": query_text,
                "response": str(response),
                "sources": sources,
                "total_time": time.time() - start_time
            }

            return result

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                "query": query_text,
                "response": "Error querying knowledge base.",
                "sources": [],
                "total_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    async def query_with_streaming(
        self,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Query the knowledge base with streaming response.
        Optimized for real-time word-by-word output to TTS.
        """
        if not self.is_initialized:
            await self.init()

        try:
            # Retrieve relevant context
            retrieval_start = time.time()
            retrieval_results = await self.retrieve_with_sources(query_text)
            results = retrieval_results.get("results", [])
            context = self.format_retrieved_context(results)
            retrieval_time = time.time() - retrieval_start

            # Format system prompt with context
            system_prompt = format_system_prompt(
                base_prompt="You are an AI assistant that answers questions based on the provided information. If the information doesn't contain the answer, acknowledge this clearly.",
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
                # Stream response in smallest possible chunks for immediate TTS processing
                streaming_response = await self.llm.astream_chat(messages)
                
                async for chunk in streaming_response:
                    # Get just the text delta/chunk - this is typically word by word or smaller
                    chunk_text = chunk.delta if hasattr(chunk, 'delta') else chunk.content
                    
                    # Only process non-empty chunks
                    if chunk_text:
                        full_response += chunk_text
                        
                        # Immediately yield each chunk for real-time TTS processing
                        # No batching or delaying of chunks
                        yield {
                            "chunk": chunk_text,
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

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        """
        doc_count = await self.index_manager.count_documents()

        return {
            "document_count": doc_count,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "llm_model": self.llm.model if self.llm else "Not initialized",
            "llm_temperature": self.llm_temperature
        }