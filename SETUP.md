# Setup Guide

## Prerequisites
- Python 3.10+
- pip or conda

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/doc_qa.git
cd doc_qa
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (Groq API key)
```

Get your Groq API key from: https://console.groq.com/

## Running the System

### Option A: FastAPI + Streamlit (Recommended for Demo)

**Terminal 1 - Backend API:**
```bash
source venv/bin/activate
python main.py
```
Backend runs on: `http://localhost:8000`

**Terminal 2 - Web UI:**
```bash
source venv/bin/activate
streamlit run streamlit_app.py
```
Web UI runs on: `http://localhost:8501`

### Option B: CLI Only
```bash
source venv/bin/activate
python -c "
from core.embeddings import get_embedder
from core.vectorstore import query_collection
from core.llm import get_llm

question = 'Your question here?'
chunks = query_collection(query_embedding=get_embedder().embed_one(question), top_k=5)
print(get_llm().answer(question, chunks))
"
```

## Uploading Your PDF

1. Start the system (see above)
2. Open http://localhost:8501
3. Upload your PDF using the file uploader
4. Ask questions about the document

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | openai | LLM service (openai/anthropic) |
| `EMBEDDING_PROVIDER` | huggingface | Embedding service |
| `OPENAI_API_KEY` | - | Groq API key (required) |
| `CHUNK_SIZE` | 500 | Text chunk size |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `VISION_ENABLED` | true | Enable vision model for diagrams |

## Troubleshooting

### "Collection documents does not exist"
- The database is empty. Upload a PDF first via Streamlit UI.

### Rate limit error from Groq
- You've hit the free tier limit (500k tokens/day)
- Wait 24 hours for reset or upgrade your Groq tier

### ChromaDB ONNX error
- This is fixed! We use pre-computed embeddings instead of ChromaDB's ONNX runtime.

## Project Structure
```
doc_qa/
├── core/              # Core RAG pipeline
│   ├── embeddings.py  # Embedding models
│   ├── ingestion.py   # PDF processing
│   ├── llm.py         # LLM interactions
│   └── vectorstore.py # ChromaDB operations
├── api/               # FastAPI backend
│   ├── models.py      # Pydantic schemas
│   └── routes/        # API endpoints
├── main.py            # FastAPI app entry
├── streamlit_app.py   # Web UI
├── config.py          # Settings management
└── requirements.txt   # Dependencies
```
