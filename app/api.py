import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from .rag import ingest_pdf, rag_query
from .utils import deterministic_id

app = FastAPI(title="KB RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=True, suffix="," + file.filename.split(".")[-1]) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        doc_id = deterministic_id(file.filename)
        count = ingest_pdf(tmp.name, doc_id=doc_id, doc_name=file.filename)
    return {"doc_id": doc_id, "chunks": count}

@app.post("/query")
async def query_endpoint(q: str = Form(...), k: int = Form(5)):
    out = rag_query(q, k=k)
    return JSONResponse(out)
