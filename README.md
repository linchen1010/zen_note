# Zen Note - AI-Powered Knowledge Search

> **Your Personal AI Assistant for Local Document Search**  
> A privacy-first, local-only RAG (Retrieval-Augmented Generation) system that lets you ask questions about your markdown notes using AI.

<div align="center">

![Zen Note Demo](https://img.shields.io/badge/Status-MVP%20Complete-brightgreen)
![Next.js](https://img.shields.io/badge/Frontend-Next.js%2015-black)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2B%20Mistral-blue)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange)

</div>

## 🎯 What is Zen Note?

Zen Note transforms your collection of markdown notes into an intelligent, searchable knowledge base. Instead of manually searching through files, simply ask questions in natural language and get AI-powered answers based on your own documents.

**🔒 Privacy-First**: Everything runs locally on your machine - no data leaves your computer.

### Key Features

- **🤖 Local AI**: Mistral 7B running via Ollama (no API keys needed)
- **📚 Smart Search**: FAISS vector database for semantic document search
- **💬 Natural Language**: Ask questions like "What did I write about vector databases?"
- **📱 Modern UI**: Clean, responsive Next.js interface with Inter typography
- **⚡ Fast**: Sub-10 second response times for most queries
- **🔗 Source Attribution**: See exactly which documents informed each answer

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- **macOS/Linux** (Windows with WSL)
- **Python 3.8+** and **Node.js 18+**
- **8GB+ RAM** (for running local LLM)

### 1. Clone & Setup
```bash
git clone <repository-url>
cd zen_note

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Start Local LLM
```bash
# Install Ollama
brew install ollama  # macOS
# curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Start Ollama service
brew services start ollama

# Download and run Mistral model (one-time setup)
ollama pull mistral
```

### 3. Build Knowledge Index
```bash
# Add your markdown files to data/markdowns/
cp your-notes/*.md data/markdowns/

# Build searchable index
cd scripts && python build_index.py
```

### 4. Start Services
```bash
# Terminal 1: Start backend API
cd backend && python main.py

# Terminal 2: Start frontend
cd frontend && npm run dev
```

### 5. Start Asking Questions! 🎉
Open **http://localhost:3000** and ask questions about your notes:
- "What is vector similarity search?"
- "Summarize my notes about machine learning"
- "How does RAG work?"

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js UI   │───▶│   FastAPI API    │───▶│   Ollama LLM    │
│  (Port 3000)   │    │   (Port 8000)    │    │  (Port 11434)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  FAISS Vector DB │
                       │   + Embeddings   │
                       └──────────────────┘
```

### RAG Pipeline Flow
1. **User Question** → Frontend captures input
2. **Vector Search** → FAISS finds relevant document chunks  
3. **Context Building** → Assemble chunks with source attribution
4. **LLM Generation** → Mistral generates answer using context
5. **Response Display** → UI shows answer with sources

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Next.js 15 + TypeScript | Modern React UI with server-side rendering |
| **Backend** | FastAPI + Python | REST API for RAG pipeline |
| **Vector DB** | FAISS | Fast similarity search for document chunks |
| **Embeddings** | SentenceTransformers | Convert text to 384-dim vectors |
| **LLM** | Ollama + Mistral 7B | Local language model for generation |
| **Styling** | TailwindCSS + Inter | Modern, accessible design system |

## 📖 Detailed Setup Guide

### Step 1: Local LLM Setup (Ollama + Mistral)

<details>
<summary>📋 Detailed LLM Installation</summary>

**Install Ollama:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows (WSL recommended)
curl -fsSL https://ollama.ai/install.sh | sh
```

**Start Ollama Service:**
```bash
# macOS (background service)
brew services start ollama

# Linux/WSL (manual start)
ollama serve
```

**Download Mistral Model:**
```bash
# This downloads ~4GB model (one-time)
ollama pull mistral

# Test the model
ollama run mistral "Hello, how are you?"
```

**Verify Installation:**
```bash
curl http://localhost:11434/api/tags
# Should show mistral model listed
```

</details>

### Step 2: Document Processing & Vector Index

<details>
<summary>📋 Building Your Knowledge Base</summary>

**Prepare Your Documents:**
```bash
# Create markdown directory
mkdir -p data/markdowns

# Add your .md files
cp ~/Documents/notes/*.md data/markdowns/
cp ~/obsidian-vault/*.md data/markdowns/

# Verify files
ls data/markdowns/
```

**Build FAISS Index:**
```bash
cd scripts

# Build with default settings
python build_index.py

# Custom settings
python build_index.py --chunk-size 1200 --overlap 300

# Test the index
python build_index.py --test-query "What is machine learning?"
```

**Index Details:**
- **Chunks**: Documents split into 1000-character pieces with 200-char overlap
- **Embeddings**: 384-dimensional vectors using `all-MiniLM-L6-v2`
- **Storage**: `backend/vector_store.faiss` + `backend/vector_store_metadata.pkl`

</details>

### Step 3: Backend API Setup

<details>
<summary>📋 FastAPI Configuration</summary>

**Install Dependencies:**
```bash
pip install -r requirements.txt
# Key packages: fastapi, faiss-cpu, sentence-transformers, ollama
```

**Start Backend:**
```bash
cd backend

# Method 1: Using startup script
./start_server.sh

# Method 2: Direct Python
python main.py

# Method 3: Uvicorn
uvicorn main:app --host localhost --port 8000 --reload
```

**Verify Backend Health:**
```bash
curl http://localhost:8000/health | jq .
# Should show all components as "healthy"
```

**API Endpoints:**
- `GET /` - API information
- `GET /health` - System health check
- `POST /ask` - Main RAG question endpoint
- `GET /search` - Vector search debugging
- `GET /status` - Detailed system status

</details>

### Step 4: Frontend UI Setup

<details>
<summary>📋 Next.js Configuration</summary>

**Install Dependencies:**
```bash
cd frontend
npm install
# Key packages: next, react, tailwindcss, typescript
```

**Start Development Server:**
```bash
npm run dev
# Runs on http://localhost:3000
```

**Production Build:**
```bash
npm run build
npm start
```

**Frontend Features:**
- **Modern UI**: Clean design with Inter typography
- **Responsive**: Works on desktop and mobile
- **Accessible**: WCAG AA compliant with keyboard navigation
- **Real-time**: Loading states and error handling

</details>

## 💡 Usage Examples

### Basic Question Answering
```
Question: "What is vector similarity search?"
Answer: "Vector similarity search refers to finding vectors in a database 
that are 'similar' to a query vector based on distance metrics like cosine 
similarity, Euclidean distance, and dot product... (Source: vector_databases.md)"
```

### Document Summarization
```
Question: "Summarize my notes about RAG systems"
Answer: "Based on your notes, RAG (Retrieval-Augmented Generation) combines 
large language models with external knowledge retrieval systems to provide 
accurate, contextual responses... (Sources: rag_overview.md, sample_knowledge.md)"
```

### Technical Comparisons
```
Question: "Compare different vector databases mentioned in my notes"
Answer: "Your notes mention several vector databases: FAISS for local similarity 
search, Pinecone for cloud-based solutions, Weaviate for graph relationships... 
(Source: vector_databases.md)"
```

## 🔧 Configuration & Customization

### Environment Variables

Create `.env` files for custom configuration:

**Backend (`.env`):**
```bash
# LLM Settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral

# API Settings
BACKEND_HOST=localhost
BACKEND_PORT=8000

# Vector Store
VECTOR_STORE_PATH=vector_store.faiss
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
```

**Frontend (`frontend/.env.local`):**
```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="Zen Note"
```

### RAG Parameter Tuning

**Retrieval Settings (`backend/rag.py`):**
```python
# Adjust these for your use case
max_chunks = 5           # Number of chunks to retrieve
score_threshold = 0.3    # Minimum similarity score (0.0-1.0)
max_context_length = 4000  # Maximum context characters
```

**LLM Settings (`backend/model_runner.py`):**
```python
# Generation parameters
max_tokens = 2000        # Response length limit
temperature = 0.7        # Creativity vs consistency (0.0-2.0)
top_p = 0.9             # Nucleus sampling
```

### Document Processing (`scripts/build_index.py`):
```python
# Chunking strategy
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks

# Custom embedding model
EMBEDDINGS_MODEL = "all-mpnet-base-v2"  # Higher quality, slower
```

## 🔍 Troubleshooting

### Common Issues & Solutions

<details>
<summary>🚨 "Ollama service not available"</summary>

**Problem**: Backend can't connect to Ollama
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
brew services start ollama  # macOS
ollama serve                 # Linux

# Check for port conflicts
lsof -i :11434
```

</details>

<details>
<summary>🚨 "Vector store not found"</summary>

**Problem**: FAISS index hasn't been built
```bash
# Build the index
cd scripts && python build_index.py

# Check if files exist
ls backend/vector_store.*

# Verify markdown files
ls data/markdowns/
```

</details>

<details>
<summary>🚨 "CORS errors in browser"</summary>

**Problem**: Frontend can't connect to backend
```bash
# Check backend CORS settings in backend/main.py
# Should include: "http://localhost:3000"

# Verify backend is running
curl http://localhost:8000/health

# Check frontend URL matches CORS origins
```

</details>

<details>
<summary>🚨 "Slow response times (>30 seconds)"</summary>

**Solutions**:
- Reduce `max_chunks` parameter (try 3 instead of 5)
- Increase `score_threshold` (try 0.5 instead of 0.3)
- Use smaller embedding model: `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`
- Check system resources: `htop` or Activity Monitor

</details>

<details>
<summary>🚨 "Poor answer quality"</summary>

**Optimization Tips**:
- **Better Chunking**: Reduce chunk size to 800 chars for more precise retrieval
- **More Context**: Increase `max_chunks` to 7-10 for complex questions
- **Lower Threshold**: Set `score_threshold` to 0.2 for broader retrieval
- **Better Documents**: Ensure markdown files are well-structured with clear headings

</details>

## 📊 System Performance

### Resource Usage
- **Memory**: ~6GB total (4GB for Mistral LLM + 2GB for system)
- **Storage**: ~4GB for Mistral model + minimal for index
- **CPU**: Moderate during generation, low during idle

### Response Times
- **Simple Questions**: 3-8 seconds
- **Complex Questions**: 8-15 seconds  
- **Vector Search Only**: <1 second

### Scaling Considerations
- **Documents**: Current setup handles ~1000 markdown files efficiently
- **Concurrent Users**: Single-user system (local only)
- **Performance**: Linear scaling with document count

## 🛣️ Roadmap & Future Features

### Version 1.1 (Planned)
- [ ] **Chat History**: Save and restore previous conversations
- [ ] **Source Highlighting**: Show specific text passages used in answers
- [ ] **Document Upload**: Drag-and-drop interface for adding new files
- [ ] **Advanced Filtering**: Filter by document type, date, or tags

### Version 1.2 (Future)
- [ ] **Multi-format Support**: PDF, DOCX, web pages
- [ ] **Real-time Indexing**: Auto-update index when files change
- [ ] **Export Features**: Save conversations as markdown/PDF
- [ ] **Custom Models**: Support for other local LLMs (Llama, CodeLlama)

### Version 2.0 (Vision)
- [ ] **Multi-user Support**: Share knowledge bases across teams
- [ ] **Cloud Deployment**: Optional cloud hosting while maintaining privacy
- [ ] **Advanced Analytics**: Usage patterns and knowledge gaps
- [ ] **Integration APIs**: Connect with Obsidian, Notion, Roam Research

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd zen_note

# Create virtual environment
python -m venv zen_note_env
source zen_note_env/bin/activate  # Linux/Mac
# zen_note_env\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Run tests
python -m pytest backend/tests/
cd frontend && npm test && cd ..
```

### Project Structure
```
zen_note/
├── backend/           # FastAPI application
│   ├── main.py       # API endpoints
│   ├── rag.py        # RAG pipeline logic
│   ├── model_runner.py # LLM integration
│   └── config.py     # Configuration
├── frontend/         # Next.js application  
│   └── src/app/      # React components
├── scripts/          # Document processing
│   └── build_index.py # Index building
├── data/            # Document storage
│   └── markdowns/   # Your markdown files
└── docs/            # Additional documentation
```

### Contribution Guidelines
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Test** your changes thoroughly
4. **Submit** a pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Ollama](https://ollama.ai/)** - For making local LLMs accessible
- **[FAISS](https://github.com/facebookresearch/faiss)** - For efficient vector similarity search
- **[Sentence Transformers](https://www.sbert.net/)** - For high-quality embeddings
- **[FastAPI](https://fastapi.tiangolo.com/)** - For the excellent Python web framework
- **[Next.js](https://nextjs.org/)** - For the modern React framework

## 📞 Support

- **Documentation**: Check component-specific READMEs in `backend/` and `frontend/` directories
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share ideas and ask questions in GitHub Discussions

---

<div align="center">

**Ready to transform your notes into an AI-powered knowledge base?**

[Get Started](#-quick-start-5-minutes) • [View Architecture](#️-system-architecture) • [Join Community](https://github.com/your-repo/discussions)

</div> 