from fastapi import FastAPI
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.ingest import router as ingest_router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Finance Clause Intelligence RAG System",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(query_router)
app.include_router(ingest_router)

@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} API is running", "env": settings.ENV}
