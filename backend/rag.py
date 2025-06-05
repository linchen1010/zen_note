"""
RAG (Retrieval-Augmented Generation) implementation
"""

import faiss
import pickle
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import VECTOR_STORE_PATH, EMBEDDINGS_MODEL
from model_runner import model_runner

# Set up logging
logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system with vector search and LLM generation"""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = {}
        self.embedding_model = None
        self._load_vector_store()
        self._load_embedding_model()

    def _load_vector_store(self):
        """Load FAISS index and metadata"""
        try:
            vector_store_path = Path(VECTOR_STORE_PATH)
            metadata_path = Path(
                str(vector_store_path).replace(".faiss", "_metadata.pkl")
            )

            if not vector_store_path.exists():
                raise FileNotFoundError(f"Vector store not found: {vector_store_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")

            # Load FAISS index
            self.index = faiss.read_index(str(vector_store_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

            # Load metadata
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.metadata = data["metadata"]

            logger.info(
                f"Loaded {len(self.chunks)} chunks from {len(set(c['metadata']['filename'] for c in self.chunks))} files"
            )

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise e

    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {EMBEDDINGS_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e

    def _embed_query(self, query: str) -> np.ndarray:
        """Convert query to embedding vector"""
        try:
            embedding = self.embedding_model.encode([query])
            # Normalize for cosine similarity (since we use IndexFlatIP)
            faiss.normalize_L2(embedding)
            return embedding.astype("float32")
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise e

    def search_similar_chunks(
        self, query: str, k: int = 5, score_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Search for similar chunks in the vector store

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of relevant chunks with metadata and scores
        """
        try:
            # Convert query to embedding
            query_embedding = self._embed_query(query)

            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if (
                    idx >= 0 and score >= score_threshold
                ):  # Valid index and above threshold
                    chunk = self.chunks[idx]
                    results.append(
                        {
                            "text": chunk["text"],
                            "score": float(score),
                            "source": chunk["metadata"]["filename"],
                            "start_char": chunk["start_char"],
                            "end_char": chunk["end_char"],
                            "metadata": chunk["metadata"],
                        }
                    )

            logger.info(
                f"Found {len(results)} relevant chunks for query: '{query[:50]}...'"
            )
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _build_context_prompt(
        self, query: str, chunks: List[Dict], max_context_length: int = 4000
    ) -> str:
        """
        Build context-aware prompt for the LLM

        Args:
            query: User's question
            chunks: Retrieved relevant chunks
            max_context_length: Maximum context length in characters

        Returns:
            Formatted prompt with context
        """
        if not chunks:
            return f"""You are a helpful assistant. Please answer this question based on your knowledge:

Question: {query}

Answer:"""

        # Build context from chunks
        context_parts = []
        current_length = 0

        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Source {i}: {chunk['source']}]\n{chunk['text']}"

            if current_length + len(chunk_text) > max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "\n\n".join(context_parts)

        # Build the full prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use the context below to answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer: Please provide a comprehensive answer based on the context above. If you reference specific information, mention which source it came from."""

        return prompt

    def ask(
        self,
        question: str,
        max_chunks: int = 5,
        score_threshold: float = 0.3,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate answer

        Args:
            question: User's question
            max_chunks: Maximum chunks to retrieve
            score_threshold: Minimum similarity score for chunks
            temperature: LLM temperature setting

        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing question: '{question[:100]}...'")

            # Step 1: Retrieve relevant chunks
            relevant_chunks = self.search_similar_chunks(
                question, k=max_chunks, score_threshold=score_threshold
            )

            # Step 2: Build context prompt
            prompt = self._build_context_prompt(question, relevant_chunks)

            # Step 3: Generate response with LLM
            llm_result = model_runner.generate_response(
                prompt=prompt, temperature=temperature, max_tokens=2000
            )

            if not llm_result["success"]:
                return {
                    "success": False,
                    "error": llm_result["error"],
                    "answer": None,
                    "sources": [],
                    "metadata": {},
                }

            # Step 4: Format response
            sources = [
                {
                    "filename": chunk["source"],
                    "score": chunk["score"],
                    "text_preview": (
                        chunk["text"][:200] + "..."
                        if len(chunk["text"]) > 200
                        else chunk["text"]
                    ),
                }
                for chunk in relevant_chunks
            ]

            return {
                "success": True,
                "answer": llm_result["response"],
                "sources": sources,
                "metadata": {
                    "chunks_found": len(relevant_chunks),
                    "model_used": llm_result["model"],
                    "generation_time": llm_result["total_time"],
                    "prompt_tokens": llm_result["prompt_tokens"],
                    "completion_tokens": llm_result["completion_tokens"],
                },
                "error": None,
            }

        except Exception as e:
            error_msg = f"RAG pipeline failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "answer": None,
                "sources": [],
                "metadata": {},
            }

    def health_check(self) -> Dict:
        """Health check for the RAG system"""
        try:
            health = {"status": "healthy", "components": {}}

            # Check vector store
            if self.index is not None and len(self.chunks) > 0:
                health["components"]["vector_store"] = {
                    "status": "healthy",
                    "vectors_count": self.index.ntotal,
                    "chunks_count": len(self.chunks),
                }
            else:
                health["components"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": "Vector store not loaded",
                }
                health["status"] = "degraded"

            # Check embedding model
            if self.embedding_model is not None:
                health["components"]["embedding_model"] = {
                    "status": "healthy",
                    "model": EMBEDDINGS_MODEL,
                }
            else:
                health["components"]["embedding_model"] = {
                    "status": "unhealthy",
                    "error": "Embedding model not loaded",
                }
                health["status"] = "degraded"

            # Check LLM
            llm_health = model_runner.health_check()
            health["components"]["llm"] = llm_health
            if llm_health["status"] != "healthy":
                health["status"] = "degraded"

            return health

        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global instance
rag_system = RAGSystem()
