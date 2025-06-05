# Zen Note Backend - FastAPI RAG System

A local-first Retrieval-Augmented Generation (RAG) API built with FastAPI, FAISS, and Ollama for private document querying.

## 🏗️ Architecture Overview

```
User Question → FastAPI → RAG Pipeline → Local LLM → Response
                  ↓
            Vector Search (FAISS) → Context Retrieval → Prompt Building
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **API Server** | `main.py` | FastAPI application with endpoints |
| **RAG Engine** | `rag.py` | Vector search + LLM integration |
| **LLM Wrapper** | `model_runner.py` | Ollama communication layer |
| **Configuration** | `config.py` | Centralized settings |

## 🚀 Quick Start

### Prerequisites
- **Ollama** running with Mistral model
- **Vector store** built (see `../scripts/README.md`)
- **Python dependencies** installed

### Start the Server

```bash
# Option 1: Using startup script (recommended)
./start_server.sh

# Option 2: Direct uvicorn
uvicorn main:app --host localhost --port 8000 --reload

# Option 3: Python script
python main.py
```

### Verify Installation
```bash
curl http://localhost:8000/health
```

## 📡 API Endpoints

### 🏠 Root Endpoint
```http
GET /
```
Returns API information and status.

### 🔍 Health Check
```http
GET /health
```
Comprehensive system health check covering:
- **Vector Store**: FAISS index status and chunk count
- **Embedding Model**: SentenceTransformer availability
- **Local LLM**: Ollama service and model status

**Response Example:**
```json
{
  "status": "healthy",
  "components": {
    "vector_store": {"status": "healthy", "vectors_count": 8},
    "embedding_model": {"status": "healthy", "model": "all-MiniLM-L6-v2"},
    "llm": {"status": "healthy", "model": "mistral", "available": true}
  }
}
```

### 🧠 Ask Question (Main RAG Endpoint)
```http
POST /ask
Content-Type: application/json

{
  "question": "What is FAISS?",
  "max_chunks": 5,           // Optional: max chunks to retrieve
  "score_threshold": 0.3,    // Optional: minimum similarity score
  "temperature": 0.7         // Optional: LLM temperature
}
```

**Complete RAG Pipeline:**
1. **Query Embedding**: Convert question to 384-dim vector
2. **Vector Search**: Find similar chunks in FAISS index
3. **Context Building**: Assemble relevant chunks with metadata
4. **LLM Generation**: Generate response using local Mistral model
5. **Response Formatting**: Return answer with source attribution

**Response Example:**
```json
{
  "success": true,
  "answer": "FAISS (Facebook AI Similarity Search) is an open-source library...",
  "sources": [
    {
      "filename": "vector_databases.md",
      "score": 0.85,
      "text_preview": "FAISS is a library developed by Facebook..."
    }
  ],
  "metadata": {
    "chunks_found": 3,
    "model_used": "mistral",
    "generation_time": 2.1,
    "prompt_tokens": 450,
    "completion_tokens": 120
  }
}
```

### 🔎 Search Only (Debug Endpoint)
```http
GET /search?query=vector%20databases&k=5&score_threshold=0.3
```

Returns vector search results without LLM generation. Useful for:
- **Debugging retrieval** quality
- **Understanding** what content is being found
- **Tuning** similarity thresholds

### 📊 System Status
```http
GET /status
```

Detailed system information including configuration and health metrics.

## 🛠️ RAG Pipeline Deep Dive

### 1. Query Processing Flow

```python
# Step 1: User Question
question = "What are vector databases?"

# Step 2: Convert to embedding (384-dimensional vector)
query_embedding = embedding_model.encode([question])

# Step 3: FAISS similarity search
scores, indices = faiss_index.search(query_embedding, k=5)

# Step 4: Retrieve relevant chunks
relevant_chunks = [chunks[idx] for idx in indices if score > threshold]

# Step 5: Build context prompt
context = "\n\n".join([f"[Source: {chunk['source']}]\n{chunk['text']}" 
                       for chunk in relevant_chunks])

# Step 6: Generate response with LLM
prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
response = ollama_generate(prompt)
```

### 2. Context Assembly Strategy

**Smart Context Building:**
- **Source Attribution**: Each chunk labeled with filename
- **Length Management**: Respect token limits (4000 chars default)
- **Relevance Ordering**: Highest scoring chunks first
- **Fallback Handling**: Graceful degradation if no relevant content found

**Prompt Template:**
```
You are a helpful assistant that answers questions based on the provided context. 
Use the context below to answer the user's question. If the context doesn't contain 
enough information to answer the question, say so clearly.

Context:
[Source 1: vector_databases.md]
Vector databases are specialized database systems...

[Source 2: rag_overview.md]
Retrieval-Augmented Generation combines...

Question: What is the difference between vector databases and traditional databases?

Answer: Please provide a comprehensive answer based on the context above. 
If you reference specific information, mention which source it came from.
```

### 3. Error Handling & Resilience

**Comprehensive Error Coverage:**
- **LLM Unavailable**: Graceful degradation with clear error messages
- **Vector Store Missing**: Helpful setup instructions
- **Malformed Requests**: Pydantic validation with detailed feedback
- **Generation Timeouts**: Configurable timeouts with fallback responses

## ⚙️ Configuration

### Environment Variables
```bash
# LLM Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral

# Backend Configuration  
BACKEND_HOST=localhost
BACKEND_PORT=8000

# Vector Store
VECTOR_STORE_PATH=vector_store.faiss
EMBEDDINGS_MODEL=all-MiniLM-L6-v2

# CORS (for frontend)
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

### Tuning Parameters

**Retrieval Parameters:**
```python
# In rag.py - RAGSystem.ask()
max_chunks = 5           # Number of chunks to retrieve
score_threshold = 0.3    # Minimum cosine similarity (0.0-1.0)
max_context_length = 4000  # Maximum context characters
```

**LLM Parameters:**
```python
# In model_runner.py - OllamaModelRunner.generate_response()
max_tokens = 2000        # Maximum response length
temperature = 0.7        # Creativity vs consistency (0.0-2.0)
top_p = 0.9             # Nucleus sampling
repeat_penalty = 1.1     # Prevent repetition
```

## 🔄 Alternative Implementations

### Different Vector Stores

**Current: FAISS**
```python
# Pros: Fast, local, no dependencies
# Cons: No built-in persistence, no filtering
index = faiss.IndexFlatIP(dimension)
```

**Alternative: Chroma**
```python
# Pros: Persistence, metadata filtering, easy setup
import chromadb
client = chromadb.Client()
collection = client.create_collection("documents")
```

**Alternative: Weaviate**
```python
# Pros: Graph relationships, advanced filtering, cloud-ready
import weaviate
client = weaviate.Client("http://localhost:8080")
```

### Different LLM Integrations

**Current: Ollama HTTP API**
```python
# Pros: Simple HTTP, model management, local
response = requests.post(f"{host}/api/generate", json=payload)
```

**Alternative: llama.cpp Python bindings**
```python
# Pros: Direct integration, more control
from llama_cpp import Llama
llm = Llama(model_path="./models/mistral-7b.gguf")
```

**Alternative: Transformers library**
```python
# Pros: Huge model selection, fine-tuning
from transformers import pipeline
generator = pipeline("text-generation", model="mistral-7b")
```

### Different Embedding Models

**Current: all-MiniLM-L6-v2 (384 dim)**
```python
# Pros: Fast, lightweight, good quality
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Alternatives:**
```python
# Higher quality, larger
'all-mpnet-base-v2'  # 768 dimensions

# Domain-specific  
'multi-qa-MiniLM-L6-cos-v1'  # Optimized for Q&A

# Multilingual
'paraphrase-multilingual-MiniLM-L12-v2'

# Code-specific
'microsoft/codebert-base'
```

## 🧪 Testing & Development

### Manual Testing
```bash
# Test health
curl http://localhost:8000/health

# Test search
curl "http://localhost:8000/search?query=FAISS&k=3"

# Test RAG
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is vector similarity search?"}'
```

### Development Mode
```bash
# Auto-reload on changes
uvicorn main:app --reload --log-level debug

# View logs
tail -f logs/rag-api.log
```

### API Documentation
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔧 Troubleshooting

### Common Issues

**1. "Ollama service not available"**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Start Ollama
brew services start ollama
```

**2. "Vector store not found"**
```bash
# Build the index
cd ../scripts
python build_index.py
```

**3. "CORS errors in browser"**
- Verify CORS_ORIGINS in config.py includes your frontend URL

**4. "Slow response times"**
- Reduce max_chunks parameter
- Increase score_threshold  
- Check Ollama model size

### Performance Optimization

**Memory Usage:**
- Vector store: ~3KB per chunk
- Embedding model: ~23MB RAM
- LLM: ~4GB RAM (Mistral 7B)

**Speed Optimization:**
- Cache embeddings for repeated queries
- Use smaller embedding models for speed
- Implement async processing for batch requests

## 📚 Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama Documentation](https://ollama.ai/docs)

---

**Next Steps:** Ready for frontend integration! The API provides CORS support for React/Next.js applications. 