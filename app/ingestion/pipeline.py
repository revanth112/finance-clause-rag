"""End-to-end document ingestion pipeline.

Flow:
  1. Load documents (PDF / TXT / MD / DOCX)
  2. Clean and normalize text
  3. Markdown-aware chunking with sliding window
  4. Embed chunks using BGE-M3
  5. Index into FAISS vector store
  6. Persist index to disk
"""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Cleans raw extracted text before chunking."""

    def clean(self, text: str) -> str:
        """Apply all cleaning steps in sequence."""
        text = self._remove_control_chars(text)
        text = self._normalize_whitespace(text)
        text = self._fix_hyphenation(text)
        text = self._remove_page_artifacts(text)
        return text.strip()

    def _remove_control_chars(self, text: str) -> str:
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _fix_hyphenation(self, text: str) -> str:
        return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    def _remove_page_artifacts(self, text: str) -> str:
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text


class IngestionPipeline:
    """Orchestrates load -> clean -> chunk -> embed -> index -> persist."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        vectorstore_dir: str = "data/vectorstore",
        index_name: str = "finance_index",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_device: str = "cpu",
        batch_size: int = 32,
    ):
        self.data_dir = data_dir
        self.vectorstore_dir = vectorstore_dir
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_device = embedding_device
        self.batch_size = batch_size

        # Lazy-initialized components
        self._loader = None
        self._chunker = None
        self._embedder = None
        self._vector_store = None
        self._preprocessor = TextPreprocessor()

    # ------------------------------------------------------------------
    # Component initializers
    # ------------------------------------------------------------------

    def _get_loader(self):
        if self._loader is None:
            from app.ingestion.loader import DocumentLoader
            self._loader = DocumentLoader(data_dir=self.data_dir)
        return self._loader

    def _get_chunker(self):
        if self._chunker is None:
            from app.ingestion.chunker import MarkdownChunker
            self._chunker = MarkdownChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self._chunker

    def _get_embedder(self):
        if self._embedder is None:
            from app.core.embeddings import BGEEmbedder
            self._embedder = BGEEmbedder(
                device=self.embedding_device,
                batch_size=self.batch_size,
            )
        return self._embedder

    def _get_vector_store(self, embedding_dim: int = 1024):
        if self._vector_store is None:
            from app.retrieval.vector_store import FAISSVectorStore
            self._vector_store = FAISSVectorStore(
                embedding_dim=embedding_dim,
                store_dir=self.vectorstore_dir,
            )
        return self._vector_store

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def load_documents(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """Stage 1: Load raw documents from disk."""
        loader = self._get_loader()
        docs = loader.load_directory(directory)
        logger.info(f"Loaded {len(docs)} documents from {directory or self.data_dir}")
        return docs

    def preprocess_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2: Clean and normalize text in all documents."""
        cleaned = []
        for doc in docs:
            doc["content"] = self._preprocessor.clean(doc["content"])
            if doc["content"]:
                cleaned.append(doc)
        logger.info(f"Preprocessed {len(cleaned)} documents (removed {len(docs)-len(cleaned)} empty)")
        return cleaned

    def chunk_documents(self, docs: List[Dict[str, Any]]):
        """Stage 3: Chunk documents using markdown-aware chunker."""
        chunker = self._get_chunker()
        chunks = chunker.chunk_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
        return chunks

    def embed_and_index(self, chunks) -> int:
        """Stages 4+5: Embed chunks and insert into FAISS."""
        embedder = self._get_embedder()
        texts = [c.content for c in chunks]

        logger.info(f"Embedding {len(texts)} chunks with {embedder.model_name}...")
        embeddings = embedder.embed_documents(texts)

        vector_store = self._get_vector_store(embedding_dim=embedder.embedding_dim)

        chunk_data = [
            (c.content, embeddings[i], c.metadata)
            for i, c in enumerate(chunks)
        ]
        vector_store.add_chunks(chunk_data)
        logger.info(f"Indexed {len(chunks)} chunks into FAISS")
        return len(chunks)

    def save_index(self):
        """Stage 6: Persist the FAISS index and metadata."""
        vs = self._get_vector_store()
        vs.save(name=self.index_name)
        logger.info(f"Saved index as '{self.index_name}' in {self.vectorstore_dir}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        directory: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Execute the complete ingestion pipeline.

        Args:
            directory: Override the source directory (default: self.data_dir)
            save: Whether to persist the FAISS index after indexing

        Returns:
            Summary dict with counts and index info
        """
        logger.info("=== Starting Ingestion Pipeline ===")

        # Stage 1: Load
        docs = self.load_documents(directory)
        if not docs:
            logger.warning("No documents found. Aborting ingestion.")
            return {"status": "empty", "docs": 0, "chunks": 0}

        # Stage 2: Clean
        docs = self.preprocess_documents(docs)

        # Stage 3: Chunk
        chunks = self.chunk_documents(docs)
        if not chunks:
            logger.warning("No chunks created. Aborting ingestion.")
            return {"status": "no_chunks", "docs": len(docs), "chunks": 0}

        # Stages 4+5: Embed + Index
        n_indexed = self.embed_and_index(chunks)

        # Stage 6: Persist
        if save:
            self.save_index()

        summary = {
            "status": "success",
            "docs_loaded": len(docs),
            "chunks_created": len(chunks),
            "vectors_indexed": n_indexed,
            "index_name": self.index_name,
            "vectorstore_dir": self.vectorstore_dir,
        }
        logger.info(f"=== Ingestion Complete: {summary} ===")
        return summary

    def run_single_file(self, file_path: str, save: bool = True) -> Dict[str, Any]:
        """Ingest a single document file into the vector store."""
        loader = self._get_loader()
        doc = loader.load_file(file_path)
        doc["content"] = self._preprocessor.clean(doc["content"])

        chunker = self._get_chunker()
        chunks = chunker.chunk_document(doc)

        if not chunks:
            return {"status": "no_chunks", "file": file_path}

        self.embed_and_index(chunks)
        if save:
            self.save_index()

        return {
            "status": "success",
            "file": file_path,
            "chunks_created": len(chunks),
        }


def run_ingestion_cli():
    """CLI entry point for running ingestion from command line."""
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Finance Clause RAG - Document Ingestion")
    parser.add_argument("--data-dir", default="data/raw", help="Source documents directory")
    parser.add_argument("--store-dir", default="data/vectorstore", help="FAISS index output directory")
    parser.add_argument("--index-name", default="finance_index", help="Name for the FAISS index")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--file", default=None, help="Ingest a single file instead of directory")
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        vectorstore_dir=args.store_dir,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_device=args.device,
    )

    if args.file:
        result = pipeline.run_single_file(args.file)
    else:
        result = pipeline.run()

    print("\nIngestion Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_ingestion_cli()
