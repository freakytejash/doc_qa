import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from config import settings


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                (self.text + str(self.metadata)).encode()
            ).hexdigest()


@dataclass
class Document:
    doc_id: str
    filename: str
    chunks: list[Chunk]
    total_pages: int
    total_chars: int


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> tuple[list[str], int]:
    """Extract per-page text from PDF bytes using pdfplumber for better layout handling.
    If vision is enabled, also describes images on each page using a vision LLM.
    Implements throttling to respect API rate limits (30k TPM = ~10 pages/min = 6s delay).
    """
    import io
    import base64
    import pdfplumber
    import time

    pages = []
    vision_page_count = 0
    
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            # Extract words with positional data and reconstruct reading order
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=True,
            )
            if not words:
                # fallback to simple extraction
                text = page.extract_text() or ""
            else:
                text = " ".join(w["text"] for w in words)

            text = _clean_text(text)

            # Optionally describe images on this page using a vision model
            if settings.vision_enabled and page.images:
                # Throttle: 6 second delays to stay under 30k TPM limit (~10 pages/min)
                if vision_page_count > 0:
                    time.sleep(6)
                image_descriptions = _describe_page_images(page, page.page_number)
                if image_descriptions:
                    text = text + "\n\n[IMAGE DESCRIPTIONS]\n" + image_descriptions
                    vision_page_count += 1

            if text.strip():
                pages.append(text)

    return pages, total_pages


def _describe_page_images(page, page_number: int) -> str:
    """Render the PDF page as an image and ask a vision model to describe its diagrams.
    Implements exponential backoff for 429 (rate limit) errors."""
    import io
    import base64
    import time
    from openai import OpenAI

    max_retries = 3
    retry_delay = 1  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            # Render the full page to a PIL image at 150 DPI
            pil_image = page.to_image(resolution=150).original

            # Encode as base64 JPEG
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=settings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe all diagrams and figures on this page. CRITICAL: "
                                    "Read all TEXT LABELS carefully (RIGHT, LEFT, ENTRY, EXIT, ENTER, LEAVE, IN, OUT, etc). "
                                    "For flow diagrams: identify the actual directional labels shown, not coordinate positions. "
                                    "State which direction cows/animals ENTER and which direction they EXIT using the labels you see. "
                                    "Be precise and only report what the labels actually say."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=512,
            )
            description = response.choices[0].message.content.strip()
            return f"Page {page_number} visual content: {description}"
            
        except Exception as e:
            error_msg = str(e)
            # Check for rate limit error (429)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    print(f"⚠️ Rate limit on page {page_number}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff: 1s, 2s, 4s
                    continue
                else:
                    print(f"⚠️ Vision rate limit exceeded for page {page_number} after {max_retries} attempts")
                    return ""
            else:
                print(f"⚠️ Vision extraction failed for page {page_number}: {e}")
                return ""


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize line endings."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _extract_printed_page(text: str) -> int | None:
    """Extract the printed page number from textbook header/footer text."""
    # Search full page text for standalone 3-4 digit page numbers
    for match in re.finditer(r'(?:^|\s)(\d{3,4})(?=\s|$)', text):
        val = int(match.group(1))
        # Plausible textbook page range: 100–1999
        if 100 <= val <= 1999:
            return val
    return None


def chunk_pages(
    pages: list[str],
    filename: str,
    doc_id: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    """
    Split page text into overlapping chunks.
    Splits on sentence boundaries where possible.
    """
    chunks: list[Chunk] = []

    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = _split_text(page_text, chunk_size, chunk_overlap)
        printed_page = _extract_printed_page(page_text)
        for idx, chunk_text in enumerate(page_chunks):
            if not chunk_text.strip():
                continue
            metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num,
                "chunk_index": idx,
            }
            if printed_page is not None:
                metadata["textbook_page"] = printed_page
            chunk = Chunk(
                text=chunk_text,
                metadata=metadata,
            )
            chunks.append(chunk)

    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Greedy sentence-aware splitter.
    Tries to break at sentence boundaries ('. ') within the chunk_size limit.
    Falls back to hard split if no boundary found.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk with overlap from previous
            overlap_text = current[-overlap:] if overlap and current else ""
            current = (overlap_text + " " + sentence).strip()

    if current:
        chunks.append(current)

    return chunks


def ingest_pdf(file_bytes: bytes, filename: str) -> Document:
    """Full ingestion pipeline: bytes → Document with Chunks."""
    doc_id = hashlib.md5(file_bytes).hexdigest()[:16]
    pages, total_pages = extract_text_from_pdf(file_bytes, filename)
    total_chars = sum(len(p) for p in pages)
    chunks = chunk_pages(pages, filename, doc_id)

    return Document(
        doc_id=doc_id,
        filename=filename,
        chunks=chunks,
        total_pages=total_pages,
        total_chars=total_chars,
    )