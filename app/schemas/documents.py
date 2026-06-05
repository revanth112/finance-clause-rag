from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentRecord(BaseModel):
    doc_id: str
    file_name: str
    source_path: str
    document_type: str = "contract"
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    language: str = "en"
    status: str = "processed"
    total_pages: Optional[int] = None
