# Document Q&A: RAG + Vision

A production-ready RAG (Retrieval-Augmented Generation) system that ingests PDFs, extracts text and diagram information via vision models, and answers questions with accurate page references.

## Architecture

```
FastAPI Backend                          Streamlit Frontend
├─ POST /documents/ingest ──────────┐    ├─ Upload PDF
│   PDF → text extract              │    ├─ Chat interface
│   + optional vision descriptions  │    └─ Source citations
├─ POST /query ─────────────────────┼──→ Question
│   retrieve chunks (ChromaDB)       │    ← Answer + sources
├─ GET /documents (list)             │
├─ DELETE /documents/{id} (remove)   │
└─ GET /health (provider info)       │
```

## Stack

| Layer       | Current Implementation                            |
|-------------|---------------------------------------------------|
| Frontend    | Streamlit (interactive web UI)                   |
| API         | FastAPI + Uvicorn                               |
| LLM         | OpenAI-compatible (Groq: llama-3.1-8b-instant)  |
| Embeddings  | HuggingFace (all-MiniLM-L6-v2, local)           |
| Vision      | Groq Vision (llama-4-scout, optional)           |
| Vector DB   | ChromaDB (persistent, local HNSW indexing)     |
| PDF Parsing | pdfplumber (layout-aware) + pypdf (fallback)   |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```env
LLM_PROVIDER=openai                          # Routes to Groq
EMBEDDING_PROVIDER=huggingface               # all-MiniLM-L6-v2
OPENAI_API_KEY=gsk_0Zn5Od0...               # Groq API key
LLM_MODEL=llama-3.1-8b-instant              # Chat model
HF_EMBEDDING_MODEL=all-MiniLM-L6-v2         # Embedding model
CHROMA_PERSIST_DIR=./chroma_db              # Local vector DB
CHUNK_SIZE=500                              # Chars per chunk
CHUNK_OVERLAP=50                            # Chunk overlap
TOP_K=5                                     # Results per query
VISION_ENABLED=true                         # Enable diagram descriptions
VISION_MODEL=meta-llama/llama-4-scout-17b   # Vision model
```

### 3. Run the system

**Start backend API:**
```bash
python main.py
# API at http://localhost:8000 | Docs at http://localhost:8000/docs
```

**Start frontend (in another terminal):**
```bash
streamlit run streamlit_app.py
# UI at http://localhost:8501
```

---

## Usage

### Upload & Ingest PDF

Through Streamlit UI:
1. Click **"Upload PDF"** in the sidebar
2. Select file → ingest starts automatically
3. Shows: pages extracted, chunks created, ingestion time

Via API:
```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@document.pdf"
```

### Ask Questions

**Through Streamlit UI:**
- Type question in chat box
- Get answer with source chunks highlighted
- Click chunks to see full text + page numbers

**Via API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics on page 515?",
    "top_k": 5
  }'
```

**Response example:**
```json
{
  "question": "...",
  "answer": "Based on Chunk 1 and Chunk 2, which are on page 515...",
  "sources": [
    {
      "filename": "LifeStock Production.pdf",
      "page": 26,
      "textbook_page": 515,
      "chunk_index": 1,
      "text": "...",
      "relevance_score": 0.95
    }
  ]
}
```

---

## Configuration Reference

| Variable                | Default                                   | Description                                |
|-------------------------|-------------------------------------------|--------------------------------------------|
| `LLM_PROVIDER`          | `openai`                                  | `openai` (routes to Groq via base_url)     |
| `EMBEDDING_PROVIDER`    | `huggingface`                             | `huggingface` or `openai`                  |
| `OPENAI_API_KEY`        | (required)                                | Groq API key (`gsk_...`)                  |
| `LLM_MODEL`             | `llama-3.1-8b-instant`                    | Chat model (Groq or OpenAI)               |
| `HF_EMBEDDING_MODEL`    | `all-MiniLM-L6-v2`                        | Local embedding model (HuggingFace)       |
| `CHROMA_PERSIST_DIR`    | `./chroma_db`                             | ChromaDB persistent storage path          |
| `CHUNK_SIZE`            | `500`                                     | Max characters per chunk                   |
| `CHUNK_OVERLAP`         | `50`                                      | Character overlap between chunks          |
| `TOP_K`                 | `5`                                       | Chunks retrieved per query                |
| `VISION_ENABLED`        | `true`                                    | Extract diagram descriptions via vision   |
| `VISION_MODEL`          | `meta-llama/llama-4-scout-17b`           | Vision model for diagram analysis         |

---

## Features

### ✅ Text Extraction
- Sentence-aware chunking respects paragraph boundaries
- Metadata per chunk: doc_id, page number, chunk index

### ✅ Textbook Page Numbers
- Auto-detects printed page numbers in PDF text (e.g., "515")
- Stores as `textbook_page` metadata separate from PDF page
- LLM references both in answers: `"page 515 (pdf_page=26)"`

### ✅ Vision Model Integration
- Extracts and describes diagrams, tables, charts
- Appends descriptions to page text before chunking
- Enables accurate answers about visual content

### ✅ Semantic Search
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Cosine similarity matching via ChromaDB
- Top-K retrieval with relevance scores

### ✅ Streaming & Async
- FastAPI async endpoints
- Streamlit real-time UI updates
- Background ingestion with progress tracking

---

## Project Structure

```
doc_qa/
├── main.py                      # FastAPI server (startup, shutdown, routes)
├── config.py                    # Pydantic Settings (env vars, validation)
├── streamlit_app.py             # Streamlit UI (chat, upload, sources)
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration
├── README.md                    # This file
├── LifeStock Production.pdf     # Example PDF (ingestable)
├── chroma_db/                   # ChromaDB persistent storage
│   ├── chroma.sqlite3
│   └── [collection data]
├── core/
│   ├── ingestion.py             # PDF parsing + chunking + vision descriptions
│   ├── embeddings.py            # HuggingFace embedding wrapper
│   ├── vectorstore.py           # ChromaDB upsert/query/delete operations
│   └── llm.py                   # LLM wrapper + prompt builder (Groq/OpenAI)
└── api/
    ├── models.py                # Pydantic schemas (Request/Response)
    └── routes/
        ├── ingest.py            # POST /documents/ingest, GET, DELETE
        └── query.py             # POST /query endpoint
```

---

## How It Works

### 1. Ingestion Pipeline

```python
PDF file
  ↓
[pdfplumber] Extract text + detect pages
  ↓
[Optional Vision] Describe diagrams (if VISION_ENABLED=true)
  ↓
[Text Chunking] Split into overlapping chunks (500 char, 50 overlap)
  ↓
[Page Detection] Find textbook page numbers in text
  ↓
[Embeddings] Convert chunks to vectors (HuggingFace)
  ↓
[ChromaDB] Store with metadata (page, chunk_index, textbook_page, etc.)
```

### 2. Query Pipeline

```
User question
  ↓
[Embedding] Convert question to vector (HuggingFace)
  ↓
[Semantic Search] Query ChromaDB, get top-5 chunks by cosine similarity
  ↓
[Prompt Building] Format context: question + top chunks + page metadata
  ↓
[LLM] Generate answer grounded in chunks (Groq/OpenAI)
  ↓
[Response] Return answer + sources with relevance scores
```

---

## Troubleshooting

### ChromaDB Connection Error

If ChromaDB fails to initialize:
```bash
rm -rf ./chroma_db
# Re-ingest PDFs (will create new collection)
```

### Vision Model Timeout

If vision descriptions take too long:
- Set `VISION_ENABLED=false` to skip vision
- Or increase timeout in config.py

### Groq API Errors

Verify your API key and rate limits:
```bash
curl -H "Authorization: Bearer gsk_YOUR_KEY" \
     https://api.groq.com/openai/v1/models
```

---

## Performance Notes

- **Embedding**: ~50ms per chunk (HuggingFace, local)
- **Retrieval**: ~10ms per query (ChromaDB, cosine)
- **LLM Inference**: ~500ms-2s per query (Groq, depends on model)
- **Vision Description**: ~2-5s per page (only if VISION_ENABLED=true)

**Full PDF ingestion (60 pages):**
- Text-only: ~30 seconds
- With vision: ~5-10 minutes
