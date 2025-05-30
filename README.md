# Zen Note - Local RAG Application

A privacy-focused Retrieval-Augmented Generation (RAG) application that allows querying `.md` files using a **local LLM**. Built with Next.js, FastAPI, and Ollama for complete data privacy.

## 🌟 Overview

Zen Note is designed to be a private, local-first application that helps you query your markdown knowledge base without sending data to external services. The system uses local LLMs for generation and FAISS for fast document retrieval.

## 🏗️ Architecture

```
User ↔️ Web UI (Next.js) ↔️ Backend API (FastAPI) ↔️ Local LLM + Vector Store ↔️ Markdown Files
```

## 🚀 Quick Start

### Prerequisites

- **macOS/Linux** (Windows support coming soon)
- **Python 3.8+**
- **Node.js 18+**
- **Homebrew** (for macOS)

### 1. Local LLM Setup (✅ Completed)

#### Install Ollama

```bash
# macOS
brew install ollama

# Start Ollama service
brew services start ollama
```

#### Download and Test Model

```bash
# Download Mistral 7B model (recommended)
ollama pull mistral

# Test the model
ollama run mistral "Hello! Can you tell me about yourself?"
```

#### Stop Ollama Service

```bash
# Stop the service when not needed
brew services stop ollama

# Or kill the process directly
pkill ollama
```

## 📁 Project Structure

```
zen_note/
├── backend/                   # FastAPI backend
│   ├── config.py             # Configuration settings
│   ├── main.py               # FastAPI app with /ask endpoint
│   ├── rag.py                # RAG logic (embedding search + LLM call)
│   ├── model_runner.py       # Wrapper to call the local LLM
│   ├── embeddings.py         # Markdown parsing + embeddings
│   └── vector_store.faiss    # Saved FAISS index
├── frontend/                 # Next.js UI
├── data/markdowns/           # Your markdown knowledge base
├── scripts/                  # Utility scripts
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Development Setup

### Backend Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Environment Configuration

The application uses `backend/config.py` for configuration. Key settings:

- **OLLAMA_HOST**: `http://localhost:11434` (default)
- **OLLAMA_MODEL**: `mistral` (configurable)
- **BACKEND_PORT**: `8000`
- **FRONTEND_PORT**: `3000`

## 🔧 Task 1 Accomplishments

### ✅ Local LLM Setup Complete

1. **Model Selection**: Chose Mistral 7B for optimal balance of performance and resource usage
2. **Installation**: Successfully installed Ollama via Homebrew on macOS
3. **Verification**: Confirmed model responds correctly to queries
4. **Integration**: Verified Python HTTP API integration works
5. **Configuration**: Set up project structure and configuration management

### Technical Decisions

- **Ollama over llama.cpp**: Better macOS integration and easier model management
- **Mistral 7B**: Good performance-to-resource ratio for local inference
- **HTTP API**: Simple integration pattern for backend communication
- **Configuration Management**: Centralized config in `backend/config.py`

## 🔧 Task 2 Accomplishments

### ✅ FAISS Index Setup Complete

1. **Sample Data**: Created sample markdown files with RAG and vector database knowledge
2. **Indexing Script**: Built comprehensive `scripts/build_index.py` with chunking and embedding
3. **FAISS Integration**: Successfully created searchable vector store with 384-dimension embeddings
4. **Search Testing**: Verified semantic search works with high-quality results
5. **Dependency Management**: Updated requirements.txt with exact installed versions

### Technical Decisions

- **Sentence Transformers**: Used `all-MiniLM-L6-v2` for balanced performance and quality
- **FAISS IndexFlatIP**: Chosen for exact cosine similarity search
- **Document Chunking**: 1000 character chunks with 200 character overlap for optimal retrieval
- **Metadata Storage**: Preserved source file information for result attribution

### 2. FAISS Index Setup (✅ Completed)

#### Build Index from Markdown Files

```bash
# Build index from all markdown files in data/markdowns/
python3 scripts/build_index.py

# Build index with custom data path
python3 scripts/build_index.py --data-path /path/to/your/markdowns

# Test with custom query
python3 scripts/build_index.py --test-query "Your test question"
```

#### Index Files

- **`backend/vector_store.faiss`**: The FAISS index file
- **`backend/vector_store_metadata.pkl`**: Chunk metadata and source information

#### Search Quality

Current index performance:
- **8 chunks** from 2 sample files
- **384-dimension** embeddings (all-MiniLM-L6-v2)
- **High-quality semantic search** with relevance scoring

## 🔍 Troubleshooting

### Ollama Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
brew services restart ollama

# Check available models
ollama list
```

### Performance Tuning

For better performance on Apple Silicon:

```bash
# Set environment variables for optimal performance
export OLLAMA_FLASH_ATTENTION="1"
export OLLAMA_KV_CACHE_TYPE="q8_0"
```

## 🚦 Service Management

### Start Services

```bash
# Start Ollama
brew services start ollama

# Verify it's running
curl http://localhost:11434/api/tags
```

### Stop Services

```bash
# Stop Ollama
brew services stop ollama
```

## 📝 What's Next

- **Task 3**: Implement FastAPI backend for RAG pipeline
- **Task 4**: Create minimal UI in Next.js
- **Task 5**: Connect frontend with backend

## 🔒 Privacy & Security

- **100% Local**: No data leaves your machine
- **No External APIs**: Everything runs locally
- **Private by Design**: Your documents and queries stay private

## 📚 Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Mistral AI](https://mistral.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)

---

**Status**: Task 1 Complete ✅ | Task 2 Complete ✅ | Next: Task 3 - FastAPI Backend 