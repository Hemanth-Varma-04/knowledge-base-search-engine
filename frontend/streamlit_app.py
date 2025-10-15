# frontend/streamlit_app.py

# --- make project root importable (so `from app...` works when run from /frontend) ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -------------------------------------------------------------------------------------

import io
import tempfile
import traceback
import streamlit as st
from dotenv import load_dotenv

# Import your RAG pipeline pieces
from app.rag import ingest_pdf, rag_query
from app.utils import deterministic_id

# Load environment variables
load_dotenv()

st.set_page_config(page_title="KB RAG (Gemini + MongoDB)", layout="wide")
st.title("Knowledge-base Search Engine — RAG with Gemini + MongoDB")

# ---- Quick env sanity checks (non-blocking) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
if not GEMINI_API_KEY:
    st.warning("`GEMINI_API_KEY` not found in environment (.env). Embeddings/answers will fail until set.")
if not MONGODB_URI:
    st.warning("`MONGODB_URI` not found in environment (.env). Vector search will fail until set.")

# ---- Sidebar: Ingestion ----
with st.sidebar:
    st.header("Ingest Documents")
    files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload one or more PDFs to build/update the vector index."
    )
    chunk_size = st.slider("Chunk size (characters)", 600, 2400, 1200, 100)
    overlap = st.slider("Chunk overlap", 0, 600, 200, 50)

    if st.button("Build / Update Index", use_container_width=True):
        if not files:
            st.error("Please upload at least one PDF.")
        else:
            with st.status("Indexing…", expanded=True) as status:
                total_chunks = 0
                try:
                    for f in files:
                        st.write(f"Processing **{f.name}** …")
                        data = io.BytesIO(f.read())

                        # Cross-platform temp file (Windows/macOS/Linux)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(data.getbuffer())
                            tmp_path = tmp.name

                        count = ingest_pdf(
                            tmp_path,
                            doc_id=deterministic_id(f.name),
                            doc_name=f.name,
                            chunk_size=chunk_size,
                            overlap=overlap,
                        )
                        total_chunks += count
                        st.write(f"→ {count} chunks added/updated")

                    status.update(
                        label=f"Indexed {len(files)} file(s), {total_chunks} chunks total in this run.",
                        state="complete"
                    )
                except Exception:
                    status.update(label="Indexing failed", state="error")
                    st.error("Something went wrong while indexing. See details below.")
                    st.code(traceback.format_exc())

# ---- Main: Query area ----
st.subheader("Ask your knowledge base")
q = st.text_input("Your question", placeholder="Ask about the uploaded documents…")
k = st.slider("Top-K passages", 3, 15, 5)

if "last_out" not in st.session_state:
    st.session_state.last_out = None

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Search & Synthesize", type="primary", use_container_width=True):
        if not q.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking with Gemini + Vector Search…"):
                try:
                    st.session_state.last_out = rag_query(q, k=k)
                except Exception:
                    st.error("Something went wrong while answering. See details below.")
                    st.code(traceback.format_exc())

    # Render answer (latest)
    if st.session_state.last_out:
        st.markdown("### Answer")
        st.write(st.session_state.last_out.get("answer", ""))

with col2:
    st.markdown("### Sources")
    if st.session_state.last_out and st.session_state.last_out.get("sources"):
        for s in st.session_state.last_out["sources"]:
            doc = s.get("doc", "Doc")
            page = s.get("page", "?")
            score = s.get("score", 0.0)
            st.caption(f"{doc} — Page {page}  (score: {score:.4f})")
    else:
        st.caption("No sources yet. Ask a question!")

# ---- Footer tips ----
st.divider()
st.info(
    "Tips:\n"
    "- Re-run indexing after adding new PDFs.\n"
    "- Smaller Top-K → more precise; larger Top-K → more coverage.\n"
    "- Ensure `.env` has `GEMINI_API_KEY` and `MONGODB_URI`."
)
