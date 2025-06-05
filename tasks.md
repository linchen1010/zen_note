# MVP Task Plan

## Task 1: Set Up Local LLM
- **Objective**: Install and configure a local LLM using `ollama` or `llama.cpp`.
- **Tasks**:
  - [x] Choose a suitable LLM model (e.g., Mistral 7B, Llama 3 8B).
  - [x] Install the necessary software and dependencies.
  - [x] Verify the installation by running a simple query.
  - [x] Write README.md documentation for LLM setup and usage.
- **Outcome**: A functional local LLM ready for integration.

## Task 2: Prepare Markdown Files and Build FAISS Index
- **Objective**: Organize markdown files and create a FAISS index for efficient retrieval.
- **Tasks**:
  - [x] Collect and organize `.md` files in the `data/markdowns/` directory.
  - [x] Use the `build_index.py` script to generate embeddings.
  - [x] Store embeddings in FAISS.
  - [x] Test the FAISS index with sample queries to ensure accuracy.
  - [x] Write README.md documentation for indexing process.
- **Outcome**: A searchable FAISS index of markdown files.

## Task 3: Implement FastAPI Backend for RAG Pipeline
- **Objective**: Develop the backend API to handle queries and integrate with the LLM and FAISS.
- **Tasks**:
  - [x] Set up a FastAPI project in the `backend/` directory.
  - [x] Implement the `/ask` endpoint in `main.py`.
  - [x] Integrate the RAG logic in `rag.py` to perform document retrieval.
  - [x] Integrate LLM invocation in `rag.py`.
  - [x] Test the API with sample requests to ensure it returns expected results.
  - [x] Write README.md documentation for backend API usage.
- **Outcome**: A working FastAPI backend capable of processing queries.

### ✅ Task 3 Accomplishments

**FastAPI Backend Complete**:

1. **Core Components Implemented**:
   - `model_runner.py` - Ollama LLM integration with health checks
   - `rag.py` - Complete RAG pipeline with FAISS vector search  
   - `main.py` - FastAPI application with comprehensive endpoints

2. **API Endpoints**:
   - `GET /` - Root endpoint with API information
   - `GET /health` - System health check (vector store, embedding model, LLM)
   - `POST /ask` - Main RAG endpoint for question answering
   - `GET /search` - Vector search without LLM generation (debugging)
   - `GET /status` - Detailed system status and configuration

3. **RAG Pipeline Features**:
   - Vector similarity search using FAISS IndexFlatIP
   - Context-aware prompt building with source attribution
   - Configurable parameters (max_chunks, score_threshold, temperature)
   - Comprehensive error handling and logging
   - Source metadata preservation

4. **Technical Implementation**:
   - CORS middleware for frontend integration
   - Pydantic models for request/response validation
   - Async lifespan management with startup health checks
   - Detailed logging and error reporting
   - Global instances for efficient resource management

5. **Testing Results**:
   - ✅ All system components healthy (vector store: 8 chunks, embedding model, LLM)
   - ✅ Vector search working correctly with relevance scoring
   - ✅ RAG pipeline generating responses using local knowledge base
   - ✅ API responding correctly to all endpoint tests
   - ✅ CORS configured for frontend on ports 3000/3001

**API Configuration**:
- Host: `localhost:8000`
- CORS origins: `http://localhost:3000`, `http://127.0.0.1:3000`
- Model: Mistral 7B via Ollama
- Vector store: 8 chunks from 2 markdown files
- Embedding model: `all-MiniLM-L6-v2`

## Task 4: Create Minimal UI in Next.js
- **Objective**: Develop a simple user interface for querying markdown files.
- **Tasks**:
  - [ ] Set up a Next.js project in the `frontend/` directory.
  - [ ] Design a basic UI with an input box using TailwindCSS and shadcn.
  - [ ] Implement result display in the UI.
  - [ ] Implement the main page in `pages/index.tsx`.
  - [ ] Test the UI to ensure it interacts correctly with the backend.
  - [ ] Write README.md documentation for frontend setup and usage.
- **Outcome**: A functional UI for user interaction.

## Task 5: Connect Frontend with Backend
- **Objective**: Integrate the frontend UI with the backend API to enable full functionality.
- **Tasks**:
  - [ ] Set up API calls from the Next.js frontend to the FastAPI backend.
  - [ ] Ensure CORS is configured correctly to allow communication.
  - [ ] Test the end-to-end flow from UI input to backend processing and result display.
  - [ ] Write comprehensive README.md documentation for the complete system.
- **Outcome**: A fully integrated system ready for user testing. 