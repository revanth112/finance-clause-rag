from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.retrieval.retriever import DenseRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5


retriever = DenseRetriever()


@router.post("/build-index")
def build_index():
    return retriever.build_index_from_chunks()


@router.post("/search")
def search_chunks(request: RetrievalRequest):
    try:
        if retriever.index_service is None:
            retriever.load_index()
        return {
            "query": request.query,
            "results": retriever.retrieve(request.query, top_k=request.top_k)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))