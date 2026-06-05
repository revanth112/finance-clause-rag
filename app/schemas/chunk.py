from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    markdown_text: str
    section_title: Optional[str] = None
    clause_type: Optional[str] = None
    page_number: Optional[int] = None
    document_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    effective_date: Optional[str] = None
    vendor_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    doc_id: str
    document_type: str
    chunk_count: int
    processed_path: str
    chunks_path: str
    metadata_path: str
