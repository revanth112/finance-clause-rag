from fastapi import APIRouter
from app.schemas.query import QueryRequest
from app.schemas.response import QueryResponse

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Main query endpoint.
    Accepts a question + optional metadata filters.
    Returns grounded answer with retrieved chunks.
    Full implementation added in Part 4.
    """
    return QueryResponse(
        question=request.question,
        answer="RAG pipeline not yet implemented. Coming in Part 4.",
        sources=[],
        confidence=0.0
    )
