from abc import ABC, abstractmethod
from functools import lru_cache

from config import settings

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer the user's question using ONLY the context chunks provided.
The context may contain diagram labels, figure captions, and layout text merged together — read carefully and extract the relevant information even if the text appears jumbled.
If you can partially answer from the context, do so.
Only say "I could not find that information" if there is truly no relevant content at all.
Be concise and cite the page number when possible.
Each chunk is labeled with a textbook_page (the printed page number in the book) — use that when the user asks about a specific page number."""


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        pdf_page = chunk["metadata"].get("page", "?")
        textbook_page = chunk["metadata"].get("textbook_page")
        filename = chunk["metadata"].get("filename", "")
        page_label = f"textbook_page={textbook_page}, pdf_page={pdf_page}" if textbook_page else f"pdf_page={pdf_page}"
        context_parts.append(
            f"[Chunk {i} | {filename} | {page_label}]\n{chunk['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context_block}\n\nQuestion: {question}"


class BaseLLM(ABC):
    @abstractmethod
    def answer(self, question: str, context_chunks: list[dict]) -> str:
        """Generate an answer grounded in the provided context chunks."""


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str = settings.llm_model):
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        prompt = build_prompt(question, context_chunks)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = settings.llm_model):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url="https://api.groq.com/openai/v1" if settings.openai_api_key.startswith("gsk_") else None,
        )
        self._model = model

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        prompt = build_prompt(question, context_chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content


@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    """Singleton LLM — instantiated once, reused across requests."""
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required.")
        return AnthropicLLM()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    return OpenAILLM()


import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
query_vec = model.encode(["cow enters parlor exits direction parallel"]).tolist()

client = chromadb.PersistentClient(path='./chroma_db', settings=ChromaSettings(anonymized_telemetry=False))
col = client.get_collection('documents')
results = col.query(
    query_embeddings=query_vec,
    n_results=3,
    include=['documents', 'metadatas']
)
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f'Page {meta["page"]}:')
    print(doc)
    print('---')
