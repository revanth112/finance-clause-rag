"""app.ingestion - Document ingestion package.

Public API:
    DocumentLoader   - Load PDFs, TXT, MD, DOCX with metadata extraction
    MarkdownChunker  - Markdown-aware chunker with sliding window overlap
    Chunk            - Dataclass representing a single chunk
    IngestionPipeline - End-to-end orchestrator: load->clean->chunk->embed->index
    TextPreprocessor - Text cleaning and normalization utilities
"""

from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import MarkdownChunker, Chunk
from app.ingestion.pipeline import IngestionPipeline, TextPreprocessor

__all__ = [
    "DocumentLoader",
    "MarkdownChunker",
    "Chunk",
    "IngestionPipeline",
    "TextPreprocessor",
]
