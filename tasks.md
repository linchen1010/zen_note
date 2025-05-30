# MVP Task Plan

## Task 1: Set Up Local LLM
- **Objective**: Install and configure a local LLM using `ollama` or `llama.cpp`.
- **Tasks**:
  - [ ] Choose a suitable LLM model (e.g., Mistral 7B, Llama 3 8B).
  - [ ] Install the necessary software and dependencies.
  - [ ] Verify the installation by running a simple query.
- **Outcome**: A functional local LLM ready for integration.

## Task 2: Prepare Markdown Files and Build FAISS Index
- **Objective**: Organize markdown files and create a FAISS index for efficient retrieval.
- **Tasks**:
  - [ ] Collect and organize `.md` files in the `data/markdowns/` directory.
  - [ ] Use the `build_index.py` script to generate embeddings.
  - [ ] Store embeddings in FAISS.
  - [ ] Test the FAISS index with sample queries to ensure accuracy.
- **Outcome**: A searchable FAISS index of markdown files.

## Task 3: Implement FastAPI Backend for RAG Pipeline
- **Objective**: Develop the backend API to handle queries and integrate with the LLM and FAISS.
- **Tasks**:
  - [ ] Set up a FastAPI project in the `backend/` directory.
  - [ ] Implement the `/ask` endpoint in `main.py`.
  - [ ] Integrate the RAG logic in `rag.py` to perform document retrieval.
  - [ ] Integrate LLM invocation in `rag.py`.
  - [ ] Test the API with sample requests to ensure it returns expected results.
- **Outcome**: A working FastAPI backend capable of processing queries.

## Task 4: Create Minimal UI in Next.js
- **Objective**: Develop a simple user interface for querying markdown files.
- **Tasks**:
  - [ ] Set up a Next.js project in the `frontend/` directory.
  - [ ] Design a basic UI with an input box using TailwindCSS and shadcn.
  - [ ] Implement result display in the UI.
  - [ ] Implement the main page in `pages/index.tsx`.
  - [ ] Test the UI to ensure it interacts correctly with the backend.
- **Outcome**: A functional UI for user interaction.

## Task 5: Connect Frontend with Backend
- **Objective**: Integrate the frontend UI with the backend API to enable full functionality.
- **Tasks**:
  - [ ] Set up API calls from the Next.js frontend to the FastAPI backend.
  - [ ] Ensure CORS is configured correctly to allow communication.
  - [ ] Test the end-to-end flow from UI input to backend processing and result display.
- **Outcome**: A fully integrated system ready for user testing. 