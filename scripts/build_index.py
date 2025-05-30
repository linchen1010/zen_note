#!/usr/bin/env python3
"""
Script to build FAISS index from markdown files
"""

import os
import sys
from pathlib import Path
import markdown
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import argparse

# Add backend to path for config import
sys.path.append(str(Path(__file__).parent.parent / "backend"))
from config import (
    MARKDOWN_DATA_PATH,
    EMBEDDINGS_MODEL,
    VECTOR_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class DocumentChunker:
    """Split documents into overlapping chunks"""

    def __init__(
        self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Find a good breaking point (end of sentence or paragraph)
            if end < len(text):
                last_period = chunk_text.rfind(".")
                last_newline = chunk_text.rfind("\n\n")
                break_point = max(last_period, last_newline)

                if break_point > start + self.chunk_size // 2:
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + break_point + 1

            chunk = {
                "text": chunk_text.strip(),
                "start_char": start,
                "end_char": end,
                "metadata": metadata or {},
            }
            chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks


class MarkdownProcessor:
    """Process markdown files and extract text content"""

    def __init__(self):
        self.md = markdown.Markdown()
        self.chunker = DocumentChunker()

    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a single markdown file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Convert markdown to plain text
            plain_text = self.md.convert(content)
            # Remove HTML tags (basic cleanup)
            import re

            plain_text = re.sub(r"<[^>]+>", "", plain_text)

            # Create metadata
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_size": len(content),
            }

            # Chunk the document
            chunks = self.chunker.chunk_text(plain_text, metadata)

            print(f"✅ Processed {file_path.name}: {len(chunks)} chunks")
            return chunks

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []

    def process_directory(self, directory: Path) -> List[Dict]:
        """Process all markdown files in a directory"""
        all_chunks = []

        md_files = list(directory.glob("*.md"))
        if not md_files:
            print(f"⚠️  No markdown files found in {directory}")
            return all_chunks

        print(f"📁 Processing {len(md_files)} markdown files...")

        for file_path in md_files:
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)

        return all_chunks


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""

    def __init__(self, model_name: str = EMBEDDINGS_MODEL):
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"✅ Model loaded successfully")

    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for all chunks"""
        if not chunks:
            return np.array([])

        texts = [chunk["text"] for chunk in chunks]
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")

        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(texts, show_progress_bar=True)

        print(f"✅ Generated embeddings: {embeddings.shape}")
        return embeddings


class FAISSIndexBuilder:
    """Build and manage FAISS index"""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = {}

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]) -> faiss.Index:
        """Build FAISS index from embeddings"""
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")

        dimension = embeddings.shape[1]
        print(f"🔄 Building FAISS index with dimension {dimension}")

        # Use IndexFlatIP (Inner Product) for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add vectors to index
        self.index.add(embeddings.astype("float32"))

        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = {
            "model_name": EMBEDDINGS_MODEL,
            "chunk_count": len(chunks),
            "dimension": dimension,
            "index_type": "IndexFlatIP",
        }

        print(f"✅ FAISS index built: {self.index.ntotal} vectors")
        return self.index

    def save_index(self, index_path: str):
        """Save index and metadata to disk"""
        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save chunks and metadata
        metadata_path = index_path.replace(".faiss", "_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

        print(f"✅ Index saved to {index_path}")
        print(f"✅ Metadata saved to {metadata_path}")

    def test_search(
        self, query: str, embedding_generator: EmbeddingGenerator, k: int = 3
    ):
        """Test the index with a sample query"""
        print(f"\n🔍 Testing search with query: '{query}'")

        # Generate query embedding
        query_embedding = embedding_generator.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        print(f"📊 Search results:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid index
                chunk = self.chunks[idx]
                print(f"  {i+1}. Score: {score:.4f}")
                print(f"     Source: {chunk['metadata']['filename']}")
                print(f"     Text: {chunk['text'][:100]}...")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from markdown files"
    )
    parser.add_argument(
        "--data-path",
        default=MARKDOWN_DATA_PATH,
        help="Path to markdown files directory",
    )
    parser.add_argument(
        "--output", default=VECTOR_STORE_PATH, help="Output path for FAISS index"
    )
    parser.add_argument(
        "--test-query", default="What is RAG?", help="Test query for index verification"
    )

    args = parser.parse_args()

    print("🚀 Starting FAISS index building process")
    print("=" * 50)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Process markdown files
    processor = MarkdownProcessor()
    chunks = processor.process_directory(Path(args.data_path))

    if not chunks:
        print("❌ No chunks generated. Exiting.")
        return

    print(f"\n📊 Total chunks generated: {len(chunks)}")

    # Step 2: Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(chunks)

    # Step 3: Build FAISS index
    index_builder = FAISSIndexBuilder()
    index_builder.build_index(embeddings, chunks)

    # Step 4: Save index
    index_builder.save_index(str(output_path))

    # Step 5: Test the index
    index_builder.test_search(args.test_query, embedding_generator)

    print("\n🎉 FAISS index building complete!")
    print(f"📁 Index saved to: {output_path}")
    print(
        f"📊 Indexed {len(chunks)} chunks from {len(set(c['metadata']['filename'] for c in chunks))} files"
    )


if __name__ == "__main__":
    main()
