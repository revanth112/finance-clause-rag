from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    page_number: int
    text: str


class DocumentRecord(BaseModel):
    doc_id: str
    file_name: str
    source_path: str
    document_type: str = "unknown"
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    total_pages: Optional[int] = None
    language: str = "en"
    status: str = "processed"