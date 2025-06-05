# Zen Note Scripts - Document Processing & Indexing

Scripts for building and managing the vector search index from markdown documents.

## 📋 Overview

This directory contains the core document processing pipeline that converts your markdown files into a searchable vector database. The main script `build_index.py` handles:

1. **Document Loading** - Reading markdown files from directories
2. **Text Chunking** - Splitting documents into optimal-sized pieces
3. **Embedding Generation** - Converting text to numerical vectors
4. **Index Building** - Creating searchable FAISS index
5. **Metadata Storage** - Preserving source and context information

## 🚀 Quick Start

### Build Index from Markdown Files

```bash
# Build index from default directory (../data/markdowns/)
python build_index.py

# Build from custom directory
python build_index.py --data-path /path/to/your/markdowns

# Test with custom query
python build_index.py --test-query "What is vector similarity search?"

# Custom output location
python build_index.py --output /path/to/custom/vector_store.faiss
```

### What Gets Generated

```
backend/
├── vector_store.faiss         # FAISS index file (binary)
└── vector_store_metadata.pkl  # Chunk metadata and source info
```

## 🏗️ Architecture Deep Dive

### Processing Pipeline

```
Markdown Files → Document Loading → Text Chunking → Embedding → FAISS Index
      ↓               ↓               ↓            ↓          ↓
   *.md files    Plain text    1000-char chunks  384-dim    Searchable
   (raw docs)    (parsed)      (overlapping)     vectors    database
```

## 📚 Document Processing Components

### 1. Document Loading (`MarkdownProcessor`)

**What it does:**
```python
# Converts markdown to plain text
markdown_content = "# Title\n\nThis is **bold** text"
plain_text = "Title\n\nThis is bold text"  # HTML tags removed
```

**Current Implementation:**
- Uses Python `markdown` library for parsing
- Strips HTML tags with regex
- Preserves text structure and readability
- Handles special characters and encoding

**Alternative Approaches:**

**Option 1: LangChain Document Loaders**
```python
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

# More sophisticated parsing
loader = DirectoryLoader(
    "../data/markdowns/", 
    glob="*.md",
    loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()
```

**Option 2: Advanced Markdown Processing**
```python
import mistune
from bs4 import BeautifulSoup

# Better HTML cleaning
def clean_markdown(content):
    html = mistune.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()
```

### 2. Text Chunking Strategy (`DocumentChunker`)

**Current Implementation: Fixed-Size with Smart Boundaries**

```python
CHUNK_SIZE = 1000        # Target chunk size
CHUNK_OVERLAP = 200      # Overlap between chunks

# Smart boundary detection
def find_best_split(text, target_end):
    # Look for sentence endings
    last_period = text.rfind('.', 0, target_end)
    # Look for paragraph breaks  
    last_newline = text.rfind('\n\n', 0, target_end)
    # Choose the best boundary
    return max(last_period, last_newline)
```

**Why This Strategy:**
- ✅ **Preserves context** across chunk boundaries with overlap
- ✅ **Respects natural breaks** - doesn't cut mid-sentence
- ✅ **Consistent size** - predictable memory and performance
- ✅ **Simple and reliable** - works across different document types

**Visual Example:**
```
Document: "The quick brown fox jumps. The lazy dog sleeps. The bird flies high..."

Chunk 1 (1000 chars): "The quick brown fox jumps. The lazy dog sleeps..."
Chunk 2 (starts 800): "...lazy dog sleeps. The bird flies high..."
                         ↑ 200-char overlap preserves context
```

### Alternative Chunking Strategies

#### 🔄 **LangChain Recursive Text Splitter**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical splitting
)

chunks = splitter.split_text(document)
```

**Advantages:**
- More sophisticated boundary detection
- Hierarchical splitting (paragraphs → sentences → words)
- Built-in optimization for various document types

#### 📝 **Semantic Chunking**

```python
from langchain.text_splitter import SpacyTextSplitter

# Split by sentences using NLP
splitter = SpacyTextSplitter(chunk_size=1000)
```

#### 📋 **Markdown-Aware Chunking**

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Respect markdown structure
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
```

#### 🎯 **Topic-Based Chunking**

```python
# Advanced: Use topic modeling to find natural breaks
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def topic_based_chunking(sentences):
    # Embed sentences
    embeddings = model.encode(sentences)
    # Cluster by topic
    clusters = KMeans(n_clusters=n_topics).fit(embeddings)
    # Group sentences by topic
    return group_by_clusters(sentences, clusters)
```

### Chunking Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fixed-size (Our approach)** | Simple, predictable, fast | May break semantic units | General documents |
| **LangChain Recursive** | Smart boundaries, flexible | More complex, dependencies | Mixed content types |
| **Semantic/Sentence** | Preserves meaning | Variable sizes, slower | Academic papers |
| **Markdown-aware** | Respects structure | Markdown-specific | Technical docs |
| **Topic-based** | Coherent themes | Complex, expensive | Long documents |

## 🧠 Embedding Generation (`EmbeddingGenerator`)

### Current Model: all-MiniLM-L6-v2

**Technical Specifications:**
```python
Model: all-MiniLM-L6-v2
Size: ~23MB
Dimensions: 384
Speed: ~2000 sentences/second on CPU
Training: Trained on 1B+ sentence pairs
```

**What happens during embedding:**
```python
text = "Vector databases enable similarity search"
# ↓ Tokenization
tokens = ["Vector", "databases", "enable", "similarity", "search"]
# ↓ Neural network processing (6 transformer layers)
hidden_states = transformer_layers(tokens)  
# ↓ Pooling (mean of token embeddings)
sentence_embedding = mean_pool(hidden_states)  # Shape: (384,)
# ↓ Normalization for cosine similarity
normalized = embedding / ||embedding||  # Unit vector
```

### Model Comparison & Alternatives

| Model | Dimensions | Size | Speed | Quality | Use Case |
|-------|------------|------|-------|---------|----------|
| **all-MiniLM-L6-v2** ⭐ | 384 | 23MB | Fast | Good | General-purpose, local |
| `all-mpnet-base-v2` | 768 | 420MB | Slower | Better | Higher quality needs |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | 23MB | Fast | Q&A optimized | Question-answering |
| `all-distilroberta-v1` | 768 | 290MB | Medium | Good | Balanced |

### Domain-Specific Embedding Models

**For Code Documents:**
```python
# Microsoft CodeBERT for code similarity
model = SentenceTransformer('microsoft/codebert-base')

# Specialized for programming languages
embeddings = model.encode([
    "def fibonacci(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    "function fibonacci(n) { return n <= 1 ? n : fib(n-1) + fib(n-2); }"
])
```

**For Scientific Papers:**
```python
# SciBERT for scientific text
model = SentenceTransformer('allenai/scibert_scivocab_uncased')
```

**For Multilingual Content:**
```python
# Support for 50+ languages
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### Custom Embedding with LangChain

```python
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# More flexible embedding pipeline
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(texts, embeddings)
```

## 🔍 FAISS Index Building

### Index Type: IndexFlatIP (Inner Product)

**Why IndexFlatIP:**
```python
# Exact cosine similarity search
index = faiss.IndexFlatIP(384)  # 384 = embedding dimension

# Normalized vectors + inner product = cosine similarity
faiss.normalize_L2(embeddings)  # Convert to unit vectors
scores, indices = index.search(query_vector, k=5)
```

**Characteristics:**
- ✅ **Exact search** - no approximation
- ✅ **Perfect recall** - finds all relevant results
- ⚠️ **Linear scaling** - search time grows with index size
- ⚠️ **Memory intensive** - stores all vectors in RAM

### Alternative FAISS Index Types

**For Large Datasets (>1M vectors):**
```python
# Approximate search with IVF (Inverted File)
quantizer = faiss.IndexFlatIP(384)
index = faiss.IndexIVFFlat(quantizer, 384, 1000)  # 1000 clusters
index.train(embeddings)  # Required for IVF
```

**For Memory Efficiency:**
```python
# Product Quantization (compressed vectors)
index = faiss.IndexPQ(384, 64, 8)  # 64 sub-vectors, 8-bit quantization
# Reduces memory by ~32x with slight quality loss
```

**For Best Performance:**
```python
# HNSW - Hierarchical Navigable Small World
index = faiss.IndexHNSWFlat(384, 32)  # 32 = connectivity
# Fast approximate search with good recall
```

### Index Performance Comparison

| Index Type | Search Speed | Memory Usage | Accuracy | Build Time |
|------------|--------------|--------------|----------|------------|
| **IndexFlatIP** ⭐ | O(n) | High | 100% | Fast |
| `IndexIVFFlat` | O(log n) | Medium | ~99% | Medium |
| `IndexPQ` | O(n) | Low | ~95% | Fast |
| `IndexHNSWFlat` | O(log n) | High | ~99% | Slow |

## 🔧 Configuration & Tuning

### Chunking Parameters

```python
# In build_index.py - DocumentChunker
CHUNK_SIZE = 1000          # Optimal: 500-2000 chars
CHUNK_OVERLAP = 200        # Optimal: 10-20% of chunk_size

# Impact on retrieval quality:
# Small chunks (< 500): Loss of context, more noise  
# Large chunks (> 2000): Diluted relevance, slower processing
# No overlap: Context loss at boundaries
# Too much overlap (> 50%): Redundancy, larger index
```

### Embedding Model Selection

```python
# Trade-offs by model size:
Models = {
    "all-MiniLM-L6-v2": {
        "speed": "fast",      # 2000+ sentences/sec
        "memory": "low",      # 23MB
        "quality": "good",    # Balanced
        "use_case": "general_purpose"
    },
    "all-mpnet-base-v2": {
        "speed": "slow",      # 500 sentences/sec  
        "memory": "high",     # 420MB
        "quality": "excellent",
        "use_case": "quality_critical"
    }
}
```

### Performance Optimization

**Batch Processing:**
```python
# Process multiple chunks at once
batch_size = 32
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    embeddings = model.encode(batch)
    index.add(embeddings)
```

**Memory Management:**
```python
# For large datasets, process in chunks
def build_large_index(chunks, batch_size=1000):
    for batch in batched(chunks, batch_size):
        embeddings = generate_embeddings(batch)
        index.add(embeddings)
        # Optional: save intermediate results
        if batch_num % 10 == 0:
            save_checkpoint(index, batch_num)
```

## 🔄 Alternative Processing Pipelines

### LangChain-Based Pipeline

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Complete pipeline in LangChain
def build_langchain_index(data_path):
    # Load documents
    loader = DirectoryLoader(data_path, glob="*.md")
    documents = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and index
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# Usage
vectorstore = build_langchain_index("../data/markdowns/")
vectorstore.save_local("langchain_faiss_index")
```

### Unstructured.io Pipeline

```python
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# Advanced document parsing
def build_unstructured_pipeline(data_path):
    elements = partition(filename=data_path)
    
    # Intelligent chunking by document structure
    chunks = chunk_by_title(
        elements, 
        max_characters=1000,
        combine_text_under_n_chars=200
    )
    
    return [chunk.text for chunk in chunks]
```

## 🧪 Testing & Validation

### Index Quality Assessment

```bash
# Test with sample queries
python build_index.py --test-query "What is FAISS?"
python build_index.py --test-query "vector similarity search"
python build_index.py --test-query "embedding models comparison"
```

### Search Quality Metrics

**Relevance Scoring:**
```python
# Check retrieved chunk quality
def evaluate_search_quality(query, expected_content):
    results = search_chunks(query, k=5)
    
    for i, result in enumerate(results):
        print(f"Rank {i+1}: Score {result['score']:.3f}")
        print(f"Source: {result['source']}")
        print(f"Preview: {result['text'][:100]}...")
        print()
```

### Performance Benchmarks

```python
# Timing different operations
import time

def benchmark_operations():
    # Chunking speed
    start = time.time()
    chunks = process_documents(documents)
    print(f"Chunking: {time.time() - start:.2f}s")
    
    # Embedding speed  
    start = time.time()
    embeddings = generate_embeddings(chunks)
    print(f"Embedding: {time.time() - start:.2f}s")
    
    # Search speed
    start = time.time()
    results = search_index(query)
    print(f"Search: {time.time() - start:.3f}s")
```

## 🚀 Usage Examples

### Basic Index Building

```bash
# Simple build with defaults
python build_index.py

# Output:
# 📁 Processing 2 markdown files...
# ✅ Processed vector_databases.md: 4 chunks
# ✅ Processed rag_overview.md: 4 chunks
# 🔄 Generating embeddings for 8 chunks...
# ✅ Generated embeddings: (8, 384)
# 🔄 Building FAISS index with dimension 384
# ✅ FAISS index built: 8 vectors
# ✅ Index saved to backend/vector_store.faiss
```

### Advanced Configuration

```bash
# Custom parameters
python build_index.py \
    --data-path ~/my-docs \
    --output ~/my-index.faiss \
    --chunk-size 1500 \
    --chunk-overlap 300 \
    --model all-mpnet-base-v2
```

### Incremental Updates (Future Feature)

```bash
# Only process changed files
python build_index.py --incremental --changed-files doc1.md,doc2.md

# Watch for changes
python build_index.py --watch --daemon
```

## 🔧 Troubleshooting

### Common Issues

**1. "No chunks generated"**
- Check if markdown files exist in data directory
- Verify file permissions and encoding
- Ensure files have readable content

**2. "Model download failed"**
- Check internet connection
- Verify model name spelling
- Consider using cached models

**3. "Memory errors during embedding"**
- Reduce batch size in embedding generation
- Use smaller embedding model
- Process files individually

**4. "Poor search quality"**
- Adjust chunk size (try 800-1200 chars)
- Increase chunk overlap (20-30%)
- Consider different embedding model
- Check if content matches query domain

### Performance Optimization

**For Large Document Collections:**
```python
# Process in batches to manage memory
EMBEDDING_BATCH_SIZE = 32
CHUNK_BATCH_SIZE = 1000

# Save intermediate results
if chunk_count % 1000 == 0:
    save_checkpoint(index, metadata)
```

**For Better Quality:**
```python
# Use higher quality models
EMBEDDINGS_MODEL = "all-mpnet-base-v2"  # 768-dim, better quality

# Optimize chunking for your content type
CHUNK_SIZE = 1200  # For longer context
CHUNK_OVERLAP = 300  # More overlap for complex topics
```

## 📚 Next Steps & Improvements

### Planned Enhancements

1. **Dynamic Index Updates**
   - File watcher for automatic rebuilding
   - Incremental updates for changed files
   - Version control for index snapshots

2. **Multi-Source Support**
   - Notion integration
   - GitHub repository indexing
   - PDF document processing
   - Web content crawling

3. **Advanced Chunking**
   - Topic-aware splitting
   - Hierarchical chunking (documents → sections → paragraphs)
   - Cross-reference preservation

4. **Quality Improvements**
   - A/B testing for different strategies
   - Automatic parameter tuning
   - Search quality metrics and monitoring

---

**Ready to build your index?** Start with `python build_index.py` and customize based on your specific needs! 