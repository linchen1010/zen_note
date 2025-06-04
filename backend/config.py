"""
Configuration settings for the RAG application
"""

import os
from pathlib import Path

# Local LLM Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Backend Configuration
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Frontend Configuration
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "localhost")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))

# Vector Store Configuration
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store.faiss")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")

# Data Configuration
MARKDOWN_DATA_PATH = Path(os.getenv("MARKDOWN_DATA_PATH", "./data/markdowns"))

# API Configuration
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
