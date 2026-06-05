from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.schemas.chunk import IngestionResponse
from app.ingestion.pipeline import run_ingestion

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/document", response_model=IngestionResponse)
def ingest_document(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported currently")

    return run_ingestion(str(path))
