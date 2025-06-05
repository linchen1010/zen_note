#!/bin/bash
# Zen Note Backend Startup Script

echo "🚀 Starting Zen Note RAG API Backend..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run from the backend directory."
    exit 1
fi

# Check if Ollama is running
echo "🔍 Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama service not running. Starting Ollama..."
    if command -v brew &> /dev/null; then
        brew services start ollama
        echo "⏳ Waiting for Ollama to start..."
        sleep 5
    else
        echo "❌ Please start Ollama manually: brew services start ollama"
        exit 1
    fi
fi

# Check if vector store exists
if [ ! -f "vector_store.faiss" ]; then
    echo "❌ Error: vector_store.faiss not found."
    echo "Please build the index first: python ../scripts/build_index.py"
    exit 1
fi

echo "✅ All dependencies ready!"
echo "🌐 Starting server on http://localhost:8000"
echo "📚 API docs will be available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn main:app --host localhost --port 8000 --reload 