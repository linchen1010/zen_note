# Zen Note - Local RAG Architecture & Technical Specification

## 🌟 Objective

Build a privacy-focused Retrieval-Augmented Generation (RAG) application that allows querying `.md` files using a **local LLM**, with a modern Next.js + TailwindCSS + shadcn UI. The system is private by default, runs entirely locally, and avoids using cloud APIs.

---

## 🧱 High-Level Architecture Overview

```
User ↔️ Web UI (Next.js) ↔️ Backend API (FastAPI) ↔️ Local LLM + Vector Store ↔️ Knowledge Sources
```

### Core Components

- **Frontend (UI)**: Built with Next.js, TailwindCSS, and shadcn. Provides a simple input box to ask questions and shows the result.
- **Backend (RAG API)**: Python FastAPI server that handles the question, performs document retrieval using a local vector store (FAISS), and invokes a local LLM.
- **LLM (Local)**: Runs via Ollama with Mistral 7B model. Local inference on user's machine.
- **Embedding Generator**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed markdown chunks.
- **Vector Store**: FAISS for fast semantic search and similarity matching.
- **Document Processing**: Python script to load, chunk, and index knowledge sources.

---

## 🔍 RAG Query Processing Pipeline

### Complete Query Flow

```
User Question → Vector Search → Context Retrieval → LLM Prompt → Response
```

**Detailed Step-by-Step Process**:

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
   Returns: Most relevant text chunks from knowledge base
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
   - Original question
   - Relevant context from documents
   ↓
   Generates answer grounded in user's knowledge base
   ```

**Critical Design Principle**: 
- 🚫 **LLM doesn't query the vector DB** - vector search happens BEFORE the LLM
- ✅ **Vector search provides context, LLM provides generation**
- 🎯 **Responses are grounded in user's documents, not general training data**

---

## 🗂 Project Structure

```
zen_note/
│
├── backend/                   # FastAPI backend
│   ├── main.py                # FastAPI app with /ask endpoint
│   ├── rag.py                 # RAG logic (embedding search + LLM call)
│   ├── model_runner.py        # Wrapper to call the local LLM
│   ├── embeddings.py          # Markdown parsing + embeddings
│   ├── config.py              # Configuration settings
│   ├── vector_store.faiss     # FAISS index file
│   └── vector_store_metadata.pkl # Chunk metadata and source information
│
├── frontend/                  # Next.js UI
│   ├── pages/
│   │   └── index.tsx          # Main UI with question input + answer display
│   ├── components/            # shadcn components
│   ├── app/                   # App routing (Next.js 13+)
│   └── tailwind.config.js     # Tailwind setup
│
├── data/
│   └── markdowns/             # Folder for all your `.md` knowledge files
│
├── scripts/
│   ├── build_index.py         # Script to parse sources, generate embeddings, store FAISS
│   └── watch_and_index.py     # (Future) Dynamic index updates
│
├── config/
│   └── sources.yaml           # (Future) Multi-source configuration
│
├── tasks.md                   # Detailed task plan for MVP
├── .env                       # Local config (model path, ports, etc.)
├── requirements.txt           # Backend dependencies
├── README.md                  # User documentation and setup
└── architecture.md            # (This file) Technical architecture reference
```

---

## ⚙️ Technology Stack & Technical Decisions

### Current Implementation (v0)

|Component|Technology|Rationale|
|---|---|---|
|**LLM**|Ollama + Mistral 7B|Better macOS integration, optimal performance/resource ratio|
|**Embedding Model**|`all-MiniLM-L6-v2`|Balanced performance and quality, 384-dimension vectors|
|**Vector DB**|FAISS IndexFlatIP|Exact cosine similarity search, high performance|
|**Backend API**|FastAPI|Fast, modern Python API framework|
|**Frontend**|Next.js + TailwindCSS + shadcn|Modern React framework with excellent UI components|
|**Document Processing**|Custom chunking + sentence-transformers|1000 char chunks with 200 char overlap for optimal retrieval|

### Configuration Management

**Centralized in `backend/config.py`**:
- **OLLAMA_HOST**: `http://localhost:11434`
- **OLLAMA_MODEL**: `mistral`
- **EMBEDDINGS_MODEL**: `all-MiniLM-L6-v2`
- **CHUNK_SIZE**: 1000 characters
- **CHUNK_OVERLAP**: 200 characters

---

## 📊 Current System Status

### ✅ Completed Tasks

**Task 1 - Local LLM Setup**:
- Ollama installation and configuration
- Mistral 7B model download and testing
- HTTP API integration verified
- Service management scripts

**Task 2 - FAISS Index Setup**:
- Sample markdown knowledge base created
- Comprehensive indexing script with chunking
- 384-dimension embeddings with high-quality semantic search
- Index files: `vector_store.faiss` + `vector_store_metadata.pkl`

### Current Index Performance
- **8 chunks** from 2 sample files
- **384-dimension** embeddings
- **High-quality semantic search** with relevance scoring

---

## 🚀 Future Architecture (v1+ Requirements)

### Dynamic Index Management 🔄

**Current Limitation**: Manual rebuild required for content changes

**Proposed Dynamic System**:
- **File Watcher Service**: Automatically detect markdown file changes
- **Incremental Updates**: Only re-index changed files instead of full rebuild
- **Smart Chunking**: Detect which specific chunks changed within files
- **Background Processing**: Update index without service interruption
- **Version Control**: Track index versions with rollback capability

**Future Implementation**:
```bash
# File watcher daemon
python3 scripts/watch_and_index.py --daemon

# Incremental updates
python3 scripts/build_index.py --incremental --changed-files file1.md,file2.md

# Scheduled rebuilds
python3 scripts/build_index.py --schedule hourly
```

### Multi-Source Knowledge Integration 📚

**Current State**: Local markdown files only

**v1 Multi-Source Architecture**:
```
Data Sources → Source Adapters → Common Format → Chunking → Vector Store
     ↓              ↓              ↓           ↓         ↓
  Notion        NotionAdapter    Markdown    Chunker   FAISS
  GitHub        GitHubAdapter    Chunks      Embedder  Index
  PDFs          PDFAdapter       Metadata    Storage   Search
  Confluence    ConfluenceAdapter
  Google Docs   GoogleDocsAdapter
```

**Supported Sources (Future)**:
- **Notion**: Workspace pages and databases
- **GitHub**: Repository markdown files
- **Confluence**: Corporate wiki content
- **Google Docs**: Shared documents
- **PDFs**: Document processing and OCR
- **Web Scraping**: Website content extraction
- **Email Archives**: Conversation indexing

**Source Adapter Interface**:
```python
# Abstract base class for all source adapters
class SourceAdapter:
    def fetch_documents(self) -> List[Document]
    def detect_changes(self) -> List[str]
    def get_metadata(self, doc_id: str) -> Dict
```

**Configuration System (Future)**:
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

---

## 🏗️ Development Phases

### Phase 0 (Current) - MVP Core
- ✅ Local LLM (Ollama + Mistral)
- ✅ FAISS vector store
- ⏳ FastAPI backend
- ⏳ Next.js frontend
- ⏳ Basic RAG pipeline

### Phase 1 - Enhanced Functionality
- 🔄 Dynamic index updates
- 🧠 Conversation history
- 🔍 Source attribution in responses
- ⚙️ Model configuration UI

### Phase 2 - Multi-Source Integration
- 📚 Notion connector
- 📁 GitHub repository sync
- 📄 PDF processing
- 🌐 Web content scraping

### Phase 3 - Advanced Features
- 🔐 User authentication
- 👥 Multi-user support
- 📊 Analytics and usage tracking
- 🎛️ Advanced configuration management

---

## ✅ v0 Features (MVP)

- ✅ **Privacy-First**: 100% local processing, no cloud APIs
- ✅ **Fast Semantic Search**: FAISS-powered vector similarity
- ✅ **Local LLM**: Ollama-based inference
- ⏳ **Clean UI**: Next.js + TailwindCSS + shadcn
- ⏳ **Simple RAG**: Question → Context → Answer pipeline

## 🎯 v1 Preview Features

- 🔄 **Dynamic Indexing**: Auto-update on content changes
- 📚 **Multi-Source**: Notion, GitHub, PDF support
- 🧠 **Conversation Memory**: Follow-up question chaining
- 🔍 **Source Attribution**: Show markdown snippets in responses
- ⚙️ **Model Flexibility**: Swap LLMs via UI
- 🎛️ **Advanced Config**: Source management interface

---

## 🔒 Privacy & Security Architecture

### Data Privacy
- **100% Local Processing**: No data leaves user's machine
- **No External APIs**: All models and processing local
- **Private by Design**: Documents and queries stay private
- **No Telemetry**: No usage tracking or data collection

### Security Considerations
- **Local Network Only**: Backend API not exposed externally
- **File System Access**: Controlled access to knowledge directories
- **Model Security**: Local models, no remote inference calls

---

## 🏁 Next Development Tasks

### Immediate (Task 3-5)
1. **FastAPI Backend**: Implement RAG pipeline API
2. **Next.js Frontend**: Create minimal query interface
3. **Integration**: Connect frontend ↔ backend ↔ LLM

### Future Development
1. **Dynamic Indexing**: File watcher implementation
2. **Multi-Source**: Notion API integration
3. **UI Enhancement**: Source attribution, conversation history
4. **Performance**: Optimization for larger knowledge bases

---

## 📚 Reference Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Mistral AI](https://mistral.ai/)

---

**Status**: Local LLM ✅ | FAISS Index ✅ | FastAPI Backend ⏳ | Next.js UI ⏳ | Integration ⏳