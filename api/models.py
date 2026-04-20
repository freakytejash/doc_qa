from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    total_chunks: int
    total_chars: int
    already_existed: bool = False
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    doc_id: str | None = Field(
        default=None,
        description="Scope retrieval to a specific document. Omit to search all docs.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    filename: str
    page: int
    chunk_index: int
    text: str
    relevance_score: float = Field(description="1 - cosine_distance (higher = better)")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    llm_provider: str
    embedding_provider: str


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    message: str
