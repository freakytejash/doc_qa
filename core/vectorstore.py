from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from core.ingestion import Chunk


@lru_cache(maxsize=1)
def get_chroma_collection() -> chromadb.Collection:
    """Return (and cache) the ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """
    Upsert chunks + their embeddings into ChromaDB.
    Returns number of chunks stored.
    """
    collection = get_chroma_collection()

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def query_collection(
    query_embedding: list[float],
    top_k: int = settings.top_k,
    where: dict | None = None,
) -> list[dict]:
    """
    Retrieve top_k most similar chunks.
    Returns list of dicts: {text, metadata, distance}.
    """
    collection = get_chroma_collection()

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})

    return hits


def doc_exists(doc_id: str) -> bool:
    """Check if any chunk for a given doc_id already exists."""
    collection = get_chroma_collection()
    results = collection.get(where={"doc_id": doc_id}, limit=1)
    return len(results["ids"]) > 0


def delete_document(doc_id: str) -> int:
    """Delete all chunks belonging to a document. Returns count deleted."""
    collection = get_chroma_collection()
    results = collection.get(where={"doc_id": doc_id})
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_documents() -> list[dict]:
    """Return unique documents stored in the collection."""
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
            }
    return list(seen.values())
