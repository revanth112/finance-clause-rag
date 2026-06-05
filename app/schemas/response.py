from pydantic import BaseModel
from typing import List, Optional

class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    doc_id: str
    document_name: str
    section_title: Optional[str] = None
    clause_type: Optional[str] = None
    page_number: Optional[int] = None
    vendor_name: Optional[str] = None
    jurisdiction: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    confidence: float
    retrieval_count: Optional[int] = None
    reranked_count: Optional[int] = None
    latency_ms: Optional[float] = None
