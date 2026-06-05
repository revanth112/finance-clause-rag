from app.ingestion.loader import load_pdf
from app.ingestion.parser import parse_pages
from app.ingestion.markdown_converter import pages_to_markdown
from app.ingestion.metadata_extractor import (
    infer_document_type,
    extract_clause_type,
    extract_jurisdiction,
    extract_vendor_name,
)
from app.ingestion.chunker import chunk_markdown_document
from app.ingestion.pipeline import run_ingestion

__all__ = [
    "load_pdf",
    "parse_pages",
    "pages_to_markdown",
    "infer_document_type",
    "extract_clause_type",
    "extract_jurisdiction",
    "extract_vendor_name",
    "chunk_markdown_document",
    "run_ingestion",
]


# """app.ingestion - Document ingestion package.

# Public API:
#     DocumentLoader   - Load PDFs, TXT, MD, DOCX with metadata extraction
#     MarkdownChunker  - Markdown-aware chunker with sliding window overlap
#     Chunk            - Dataclass representing a single chunk
#     IngestionPipeline - End-to-end orchestrator: load->clean->chunk->embed->index
#     TextPreprocessor - Text cleaning and normalization utilities
# """

# from app.ingestion.loader import DocumentLoader
# from app.ingestion.chunker import MarkdownChunker, Chunk
# from app.ingestion.pipeline import IngestionPipeline, TextPreprocessor

# __all__ = [
#     "DocumentLoader",
#     "MarkdownChunker",
#     "Chunk",
#     "IngestionPipeline",
#     "TextPreprocessor",
# ]
