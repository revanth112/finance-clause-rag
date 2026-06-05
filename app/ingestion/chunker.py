import re
import uuid
from typing import List, Tuple

from app.schemas.chunk import ChunkRecord
from app.ingestion.metadata_extractor import (
    extract_clause_type,
    extract_jurisdiction,
    extract_vendor_name,
)


def split_markdown_sections(markdown_text: str) -> List[Tuple[str, str]]:
    sections = re.split(r"\n### ", markdown_text)
    results: List[Tuple[str, str]] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        parts = section.split("\n", 1)
        title = parts[0].replace("### ", "").strip()
        body = parts[1].strip() if len(parts) > 1 else ""
        results.append((title, body))

    return results


def chunk_markdown_document(
    doc_id: str,
    markdown_text: str,
    document_type: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> List[ChunkRecord]:
    sections = split_markdown_sections(markdown_text)
    chunks: List[ChunkRecord] = []
    chunk_index = 0

    for title, body in sections:
        full_text = f"{title}\n\n{body}".strip()

        if len(full_text) <= chunk_size:
            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=full_text,
                    markdown_text=full_text,
                    section_title=title,
                    clause_type=extract_clause_type(title, full_text),
                    document_type=document_type,
                    jurisdiction=extract_jurisdiction(full_text),
                    vendor_name=extract_vendor_name(full_text),
                    metadata={
                        "section_title": title,
                        "chunking_strategy": "markdown_section"
                    }
                )
            )
            chunk_index += 1
            continue

        start = 0
        while start < len(full_text):
            end = start + chunk_size
            piece = full_text[start:end]

            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=piece,
                    markdown_text=piece,
                    section_title=title,
                    clause_type=extract_clause_type(title, piece),
                    document_type=document_type,
                    jurisdiction=extract_jurisdiction(piece),
                    vendor_name=extract_vendor_name(piece),
                    metadata={
                        "section_title": title,
                        "chunking_strategy": "markdown_section_with_fallback"
                    }
                )
            )
            chunk_index += 1
            start += max(chunk_size - overlap, 1)

    return chunks



# """Markdown-aware chunker for finance documents."""
# import re
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass, field


# @dataclass
# class Chunk:
#     """A single document chunk with content and metadata."""
#     content: str
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     chunk_id: str = ""
#     section_title: str = ""
#     chunk_index: int = 0

#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "content": self.content,
#             "metadata": self.metadata,
#             "chunk_id": self.chunk_id,
#             "section_title": self.section_title,
#             "chunk_index": self.chunk_index,
#         }


# class MarkdownChunker:
#     """Splits documents using markdown heading hierarchy with overlap."""

#     # Heading patterns: # H1, ## H2, ### H3
#     HEADING_PATTERN = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
#     # Finance-specific clause markers
#     CLAUSE_PATTERN = re.compile(
#         r'^(Section|Article|Clause|Schedule|Exhibit|Appendix)\s+[\d\.]+',
#         re.MULTILINE | re.IGNORECASE
#     )

#     def __init__(
#         self,
#         chunk_size: int = 512,
#         chunk_overlap: int = 64,
#         min_chunk_size: int = 50,
#         respect_headings: bool = True,
#     ):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.min_chunk_size = min_chunk_size
#         self.respect_headings = respect_headings

#     def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
#         """Chunk a loaded document dict into Chunk objects."""
#         content = document["content"]
#         base_metadata = document.get("metadata", {})
#         doc_id = base_metadata.get("filename", "doc")

#         if self.respect_headings and self._has_markdown_structure(content):
#             raw_chunks = self._markdown_split(content)
#         else:
#             raw_chunks = self._sliding_window_split(content)

#         chunks = []
#         for idx, (text, section) in enumerate(raw_chunks):
#             if len(text.strip()) < self.min_chunk_size:
#                 continue
#             chunk_meta = {**base_metadata, "section": section, "chunk_index": idx}
#             chunk = Chunk(
#                 content=text.strip(),
#                 metadata=chunk_meta,
#                 chunk_id=f"{doc_id}::chunk_{idx}",
#                 section_title=section,
#                 chunk_index=idx,
#             )
#             chunks.append(chunk)

#         return chunks

#     def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
#         """Chunk multiple documents."""
#         all_chunks = []
#         for doc in documents:
#             all_chunks.extend(self.chunk_document(doc))
#         return all_chunks

#     def _has_markdown_structure(self, text: str) -> bool:
#         """Check if document has markdown headings or clause markers."""
#         return bool(self.HEADING_PATTERN.search(text) or self.CLAUSE_PATTERN.search(text))

#     def _markdown_split(self, text: str) -> List[tuple]:
#         """Split by markdown headings, then slide-window within each section."""
#         sections = []
#         current_section = "Introduction"
#         current_content = []

#         for line in text.split('\n'):
#             heading_match = self.HEADING_PATTERN.match(line)
#             clause_match = self.CLAUSE_PATTERN.match(line)

#             if heading_match or clause_match:
#                 # Save current section
#                 if current_content:
#                     section_text = '\n'.join(current_content).strip()
#                     if section_text:
#                         sections.append((section_text, current_section))
#                 # Start new section
#                 current_section = line.strip()
#                 current_content = [line]
#             else:
#                 current_content.append(line)

#         # Save last section
#         if current_content:
#             section_text = '\n'.join(current_content).strip()
#             if section_text:
#                 sections.append((section_text, current_section))

#         # Now apply sliding window within each section
#         result = []
#         for section_text, section_title in sections:
#             sub_chunks = self._sliding_window_split(section_text, section_title)
#             result.extend(sub_chunks)

#         return result

#     def _sliding_window_split(self, text: str, section: str = "") -> List[tuple]:
#         """Sliding window tokenization by word count."""
#         words = text.split()
#         chunks = []
#         start = 0

#         while start < len(words):
#             end = min(start + self.chunk_size, len(words))
#             chunk_words = words[start:end]
#             chunk_text = ' '.join(chunk_words)
#             chunks.append((chunk_text, section))
#             if end >= len(words):
#                 break
#             start += self.chunk_size - self.chunk_overlap

#         return chunks if chunks else [(text, section)]
