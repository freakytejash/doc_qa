from fastapi import APIRouter, HTTPException, UploadFile, File, status

from api.models import IngestResponse, DocumentListResponse, DocumentInfo, DeleteResponse
from core.embeddings import get_embedder
from core.ingestion import ingest_pdf
from core.vectorstore import upsert_chunks, doc_exists, list_documents, delete_document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF document",
)
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are supported.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    # Parse + chunk the PDF
    document = ingest_pdf(file_bytes, file.filename)

    # Skip re-ingestion if same file was already uploaded (sha256 match)
    if doc_exists(document.doc_id):
        return IngestResponse(
            doc_id=document.doc_id,
            filename=document.filename,
            total_pages=document.total_pages,
            total_chunks=len(document.chunks),
            total_chars=document.total_chars,
            already_existed=True,
            message="Document already indexed. Skipping re-ingestion.",
        )

    # Embed all chunks
    embedder = get_embedder()
    texts = [c.text for c in document.chunks]
    embeddings = embedder.embed(texts)

    # Store in ChromaDB
    upsert_chunks(document.chunks, embeddings)

    return IngestResponse(
        doc_id=document.doc_id,
        filename=document.filename,
        total_pages=document.total_pages,
        total_chunks=len(document.chunks),
        total_chars=document.total_chars,
        message=f"Successfully indexed {len(document.chunks)} chunks from {document.total_pages} pages.",
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def get_documents():
    docs = list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a document and all its chunks",
)
async def remove_document(doc_id: str):
    deleted = delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with doc_id '{doc_id}'.",
        )
    return DeleteResponse(
        doc_id=doc_id,
        chunks_deleted=deleted,
        message=f"Deleted {deleted} chunks for document '{doc_id}'.",
    )
