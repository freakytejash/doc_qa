from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm embedder and LLM on startup so first request isn't slow."""
    print(f"Starting up — LLM: {settings.llm_provider} | Embeddings: {settings.embedding_provider}")
    from core.embeddings import get_embedder
    from core.llm import get_llm
    get_embedder()
    get_llm()
    print("Models loaded. Ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Document Q&A API",
    description=(
        "Ingest PDF documents and ask questions about them using RAG "
        "(Retrieval-Augmented Generation).\n\n"
        "**Flow:** Upload PDF → chunks stored in ChromaDB → "
        "ask a question → relevant chunks retrieved → LLM answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.active_embedding_model,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
