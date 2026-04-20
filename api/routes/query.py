from fastapi import APIRouter, HTTPException, status

from api.models import QueryRequest, QueryResponse, SourceChunk
from config import settings
from core.embeddings import get_embedder
from core.llm import get_llm
from core.vectorstore import query_collection, get_chroma_collection

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Ask a question about ingested documents",
)
async def query_documents(body: QueryRequest):
    collection = get_chroma_collection()
    if collection.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please ingest a document first.",
        )

    # Embed the question
    embedder = get_embedder()
    query_embedding = embedder.embed_one(body.question)

    # Retrieve relevant chunks (optionally scoped to one doc)
    where_filter = {"doc_id": body.doc_id} if body.doc_id else None
    chunks = query_collection(
        query_embedding=query_embedding,
        top_k=body.top_k,
        where=where_filter,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks found. Try rephrasing your question.",
        )

    # Generate answer via LLM
    llm = get_llm()
    answer = llm.answer(body.question, chunks)

    # Build source attribution
    sources = [
        SourceChunk(
            filename=c["metadata"].get("filename", ""),
            page=c["metadata"].get("page", 0),
            chunk_index=c["metadata"].get("chunk_index", 0),
            text=c["text"],
            relevance_score=round(1 - c["distance"], 4),
        )
        for c in chunks
    ]

    return QueryResponse(
        question=body.question,
        answer=answer,
        sources=sources,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
    )
