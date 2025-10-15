import hashlib
import re
from typing import Iterable, List

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(txt: str) -> str:
    # normalize whitespace
    return WHITESPACE_RE.sub(" ", txt).strip()


def deterministic_id(*parts: Iterable[str]) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
    return m.hexdigest()[:16]


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Simple, robust character-based chunking with overlap.
    Tuned for Gemini 1.5 context usage.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
