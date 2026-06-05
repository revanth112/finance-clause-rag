from pydantic import BaseModel
from typing import Optional, List

class MetadataFilters(BaseModel):
    document_type: Optional[str] = None
    vendor_name: Optional[str] = None
    clause_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    contract_year: Optional[str] = None
    risk_level: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    filters: Optional[MetadataFilters] = None
    top_k: Optional[int] = 5

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the termination notice period?",
                "filters": {
                    "clause_type": "termination",
                    "vendor_name": "Vendor A",
                    "jurisdiction": "India"
                },
                "top_k": 5
            }
        }
