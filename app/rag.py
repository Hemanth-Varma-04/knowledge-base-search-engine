# app/rag.py

import os
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from pymongo.collection import Collection
from pymongo import UpdateOne

from .db import get_collection
from .utils import chunk_text, deterministic_id, clean_text
from .prompts import SYSTEM_RAG_INSTRUCTIONS, USER_TEMPLATE

# --- Environment (no hard failures at import time) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
GEN_MODEL   = os.getenv("GEMINI_GEN_MODEL",   "gemini-2.5-flash")

# Configure Google client only if we have a key (safe to call multiple times)
def _ensure_gemini_configured():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY missing. Add it to your .env before running embeddings or generation."
        )
    # idempotent; safe if called multiple times
    genai.configure(api_key=GEMINI_API_KEY)

# Create a model handle (newer SDKs accept just the model name in the ctor)
def _get_generative_model():
    _ensure_gemini_configured()
    try:
        return genai.GenerativeModel(GEN_MODEL)
    except TypeError:
        # Extreme fallback if API changes again
        return genai.GenerativeModel(str(GEN_MODEL))


# ---------------- Embeddings ----------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts with Gemini embeddings (returns 768-dim vectors)."""
    _ensure_gemini_configured()
    out: List[List[float]] = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t)
        out.append(r["embedding"])
    return out


# ---------------- Ingestion / Upsert ----------------
def upsert_chunks(
    doc_id: str,
    doc_name: str,
    chunks: List[Tuple[int, str]],
    col: Collection = None
) -> int:
    """Upsert chunk docs (with embeddings) into MongoDB using UpdateOne."""
    col = col or get_collection()
    if not chunks:
        return 0

    texts = [c[1] for c in chunks]
    vecs  = embed_texts(texts)

    ops: List[UpdateOne] = []
    for (page, text), emb in zip(chunks, vecs):
        chunk_id = deterministic_id(doc_id, str(page), text[:64])
        filt = {"doc_id": doc_id, "chunk_id": chunk_id}
        update = {
            "$set": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "embedding": emb,
                "metadata": {"doc_name": doc_name, "page": page},
            }
        }
        ops.append(UpdateOne(filt, update, upsert=True))

    if ops:
        col.bulk_write(ops, ordered=False)
    return len(ops)


def ingest_pdf(
    pdf_path: str,
    doc_id: str,
    doc_name: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> int:
    """Extract per-page text, chunk, embed, and upsert into MongoDB."""
    from .pdf_loader import extract_pdf_text

    pages = extract_pdf_text(pdf_path)
    all_chunks: List[Tuple[int, str]] = []
    for p in pages:
        txt = p.get("text") or ""
        if not txt:
            continue
        for ch in chunk_text(txt, chunk_size=chunk_size, overlap=overlap):
            all_chunks.append((p["page"], ch))

    return upsert_chunks(doc_id, doc_name, all_chunks)


# ---------------- Retrieval ----------------
def vector_search(query: str, k: int = 5, col: Collection = None) -> List[Dict]:
    """ANN retrieval via MongoDB Atlas Vector Search ($vectorSearch)."""
    col = col or get_collection()
    qvec = embed_texts([clean_text(query)])[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "default",            # change if your index name differs
                "path": "embedding",
                "queryVector": qvec,
                "numCandidates": max(50, k * 10),
                "limit": k,
            }
        },
        {"$project": {"text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(col.aggregate(pipeline))


def build_context(results: List[Dict]) -> str:
    ctx_parts = []
    for r in results:
        meta = r.get("metadata", {})
        name = meta.get("doc_name", "Doc")
        page = meta.get("page", "?")
        header = f"[Source: {name}, Page {page}]\n"
        ctx_parts.append(header + (r.get("text") or ""))
    return "\n\n".join(ctx_parts)


# ---------------- Synthesis ----------------
def synthesize_answer(question: str, results: List[Dict]) -> Dict:
    """Build a single-turn RAG prompt and generate an answer with Gemini."""
    model = _get_generative_model()

    context = build_context(results)
    prompt = (
        SYSTEM_RAG_INSTRUCTIONS
        + "\n\nUsing ONLY the context below, answer succinctly.\n\n"
        + f"# Question\n{question}\n\n# Context\n{context}"
    )

    resp = model.generate_content(
        [prompt],  # one combined user turn; no 'system'/'assistant' roles
        generation_config={"temperature": 0.2, "top_p": 0.9},
    )

    return {
        "answer": getattr(resp, "text", "") or "",
        "sources": [
            {
                "doc": r.get("metadata", {}).get("doc_name"),
                "page": r.get("metadata", {}).get("page"),
                "score": float(r.get("score", 0.0)),
            }
            for r in results
        ],
    }


def rag_query(question: str, k: int = 5) -> Dict:
    hits = vector_search(question, k=k)
    return synthesize_answer(question, hits)
