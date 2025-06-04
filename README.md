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

#### Manual Index Rebuild Process

When you add new markdown files or update existing ones, you need to rebuild the index:

```bash
# Rebuild index after content changes
python3 scripts/build_index.py

# The script will:
# 1. Process all .md files in data/markdowns/
# 2. Split documents into overlapping chunks (~1000 chars)
# 3. Generate 384-dimension embeddings for each chunk
# 4. Build FAISS index with cosine similarity search
# 5. Save index + metadata to backend/ directory
```

**Important**: Always rebuild the index when:
- ✅ Adding new markdown files
- ✅ Updating existing markdown content
- ✅ Changing chunking parameters
- ✅ Switching embedding models

## 🚀 Future Enhancements (v1 Requirements)

### Dynamic Index Updates 🔄

**Current State**: Manual rebuild required for any content changes

**Proposed Dynamic System**:
- **File Watcher**: Automatically detect changes to markdown files
- **Incremental Updates**: Only re-index changed files instead of full rebuild
- **Smart Chunking**: Detect which specific chunks changed within a file
- **Background Processing**: Update index in background without service interruption
- **Version Control**: Track index versions and rollback capability

**Implementation Ideas**:
```bash
# File watcher service (future implementation)
python3 scripts/watch_and_index.py --daemon

# Incremental update (future implementation)  
python3 scripts/build_index.py --incremental --changed-files file1.md,file2.md

# Scheduled rebuilds (future implementation)
python3 scripts/build_index.py --schedule hourly
```

### Flexible Input Sources 📚

**Current State**: Only supports local markdown files

**v1 Multi-Source Support**:
- **Notion Integration**: Sync Notion pages and databases
- **Google Docs**: Import shared documents
- **Confluence**: Corporate wiki integration
- **GitHub**: Pull markdown files from repositories
- **Web Scraping**: Extract content from websites
- **PDF Processing**: Convert PDFs to searchable text
- **Email Archives**: Index email conversations

**Proposed Architecture**:
```
Data Sources → Source Adapters → Common Format → Chunking → Vector Store
     ↓              ↓              ↓           ↓         ↓
  Notion        NotionAdapter    Markdown    Chunker   FAISS
  GitHub        GitHubAdapter    Chunks      Embedder  Index
  PDFs          PDFAdapter       Metadata    Storage   Search
```

**Configuration Example** (future):
```yaml
# config/sources.yaml
sources:
  - type: "notion"
    workspace_id: "your-workspace"
    auth_token: "env:NOTION_TOKEN"
  - type: "github"
    repo: "user/knowledge-repo"
    path: "docs/"
  - type: "local"
    path: "data/markdowns/"
```

### Source Adapter Interface 🔌

**Future Implementation**:
```python
# Abstract base class for all source adapters
class SourceAdapter:
    def fetch_documents(self) -> List[Document]
    def detect_changes(self) -> List[str]
    def get_metadata(self, doc_id: str) -> Dict
```

## 🔍 How RAG Queries Work

**Your Question**: "When we do the query, what happened? Are we asking the LLM and LLM is querying our vector database?"

**Answer**: Great question! Here's the exact flow:

### Query Processing Pipeline 🔄

```
User Question → Vector Search → Context Retrieval → LLM Prompt → Response
```

**Step-by-Step Process**:

1. **🔍 Query Vectorization**
   ```
   User: "What is RAG?"
   ↓
   Embedding Model converts question to 384-dim vector
   ```

2. **🎯 Vector Similarity Search**
   ```
   Query Vector → FAISS Index → Top K Similar Chunks
   ↓
   Returns: Most relevant text chunks from your markdown files
   ```

3. **📝 Context Assembly**
   ```
   System assembles prompt:
   "Based on this context: [retrieved chunks]
   Answer the user's question: What is RAG?"
   ```

4. **🤖 LLM Generation**
   ```
   Ollama (Mistral) receives:
   - Your original question
   - Relevant context from YOUR documents
   ↓
   Generates answer based on YOUR knowledge base
   ```

**Key Points**:
- 🚫 **LLM doesn't query the vector DB** - that happens BEFORE the LLM
- ✅ **Vector search finds relevant context first**
- ✅ **LLM uses that context to answer your question**
- 🎯 **This ensures answers are based on YOUR documents, not general training data**

**Example Flow**:
```
Question: "What vector databases are mentioned?"
↓
Vector Search finds chunks mentioning: "FAISS, Pinecone, Weaviate, Chroma, Qdrant"
↓
LLM receives context + question
↓
Response: "Based on your documents, the vector databases mentioned are FAISS, Pinecone, Weaviate, Chroma, and Qdrant..."
```

This is why RAG is so powerful - it grounds the LLM's responses in your specific knowledge! 🚀

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

**Status**: Task 1 Complete ✅ | Task 2 Complete ✅ | Task 3 Complete ✅ | Next: Task 4 - Next.js Frontend 

## 🔧 Task 3 Accomplishments

### ✅ FastAPI Backend Complete

1. **Core API Implementation**: Built comprehensive FastAPI backend with RAG pipeline
2. **Endpoint Development**: Implemented `/ask`, `/health`, `/search`, and `/status` endpoints
3. **Integration Testing**: Verified all components work together seamlessly
4. **Error Handling**: Added robust error handling and logging throughout
5. **CORS Configuration**: Set up CORS for frontend integration

### Technical Implementation

- **FastAPI Application**: Modern async API with Pydantic validation
- **RAG Pipeline**: Complete retrieval-augmented generation system
- **LLM Integration**: Ollama wrapper with health monitoring
- **Vector Search**: FAISS-powered semantic search
- **Documentation**: Auto-generated OpenAPI docs at `/docs`

### 3. FastAPI Backend Setup (✅ Completed)

#### Start the Backend Server

```bash
# Option 1: Using the startup script (recommended)
cd backend
./start_server.sh

# Option 2: Manual startup
cd backend
uvicorn main:app --host localhost --port 8000 --reload
```

#### API Endpoints

The server runs on `http://localhost:8000` with the following endpoints:

- **`GET /`** - API information and status
- **`POST /ask`** - Main RAG endpoint for asking questions
- **`GET /health`** - System health check
- **`GET /search`** - Vector search without LLM generation
- **`GET /status`** - Detailed system status and configuration
- **`GET /docs`** - Interactive API documentation

#### Test the API

```bash
# Test basic functionality
curl http://localhost:8000/

# Check system health
curl http://localhost:8000/health

# Test vector search
curl "http://localhost:8000/search?query=vector%20databases"

# Ask a question (RAG pipeline)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FAISS?"}'
```

#### API Response Format

The `/ask` endpoint returns:
```json
{
  "success": true,
  "answer": "Answer based on your knowledge base...",
  "sources": [
    {
      "filename": "vector_databases.md",
      "score": 0.85,
      "text_preview": "Relevant chunk preview..."
    }
  ],
  "metadata": {
    "chunks_found": 3,
    "model_used": "mistral",
    "generation_time": 2.1
  }
}
``` 