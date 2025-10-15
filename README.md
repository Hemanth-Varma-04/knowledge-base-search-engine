# Knowledge Base Search Engine using Gemini API and MongoDB Vector Search with Streamlit Interface

## Overview
This project is a knowledge base search engine that allows users to upload one or more PDF documents and ask questions about their content. The system extracts and chunks text from the PDFs, creates vector embeddings using the Gemini API, stores them in MongoDB Atlas Vector Search, and answers user queries using Retrieval Augmented Generation (RAG) with the Gemini language model.

The main goal is to provide an end to end searchable knowledge base built on top of your own documents with accurate retrieval and AI synthesized answers.
##
# Demo Video

https://github.com/user-attachments/assets/b9eded98-450a-4651-bcd6-a1b55c795725

## How it Works
1. The user uploads one or more PDF files through the Streamlit interface.
2. The system extracts text from each page using pdfplumber and cleans the text.
3. The text is chunked into fixed length segments with overlapping tokens for context preservation.
4. Each chunk is converted into an embedding vector using the Gemini text embedding model (text embedding 004).
5. All chunks are stored in a MongoDB collection with their text, metadata, and vector embeddings.
6. When a query is submitted the system embeds the query using Gemini and runs a vector search in MongoDB to find the most similar chunks.
7. The top K retrieved chunks are combined into a context block.
8. The context and query are passed to a Gemini generative model (gemini 1.5 flash or pro) to produce a concise and citation backed answer.
9. The Streamlit interface displays the final synthesized answer and the list of sources with document name and page number.
## Project Structure
kb rag streamlit
│
├── README.md
├── requirements.txt
├── .env.example
├── scripts
│   └── create mongo vector index.json
│
├── data
│   ├── raw  contains uploaded pdfs
│   └── processed  contains extracted text jsonl
│
├── app
│   ├── __init__.py
│   ├── api.py  FastAPI endpoints for ingest and query
│   ├── rag.py  core pipeline for embedding retrieval and generation
│   ├── db.py  MongoDB connection and helper methods
│   ├── pdf loader.py  text extraction from pdf files
│   ├── prompts.py  RAG prompt templates
│   └── utils.py  text cleaning chunking and hashing utilities
│
└── frontend
    └── streamlit app.py  Streamlit UI for uploading files and querying

## Environment Setup
1. Clone the repository and open it in your code editor.
2. Create a virtual environment and install dependencies:
   pip install -r requirements.txt
3. Copy the .env.example file to .env and fill in the required values:
   GEMINI_API_KEY=your gemini api key
   MONGODB_URI=your mongodb atlas uri
   MONGODB_DB=kb_rag
   MONGODB_COL=chunks
   GEMINI_EMBED_MODEL=text-embedding-004
   GEMINI_GEN_MODEL=gemini-1.5-flash
4. Create a Vector Search index in MongoDB Atlas using the script in scripts/create mongo vector index.json.

Running the Application
Option 1: Using Streamlit only
   streamlit run frontend/streamlit app.py

Option 2: Using FastAPI backend and Streamlit frontend
   uvicorn app.api:app --reload --port 8000

Streamlit Usage Steps
1. Launch Streamlit and open the local web interface.
2. Upload one or more PDF documents in the sidebar.
3. Click on Build or Update Index.
4. Enter a question and click on Search and Synthesize.
5. View the answer and sources in the UI.

Technical Workflow
PDF text extraction -> text cleaning and chunking -> embedding creation -> data storage -> query retrieval -> context building -> answer synthesis -> output.

Requirements
python-dotenv
pymongo
google-generativeai
fastapi
uvicorn
streamlit
pdfplumber
pypdf
numpy
pandas
tqdm

Evaluation Criteria
Retrieval accuracy
Synthesis quality
Code structure
Integration correctness
UI usability

Security Recommendations
Never commit .env files.
Restrict MongoDB IP access.
Use environment variables for secrets.

Testing
Use sample PDFs for smoke testing.
Check chunk count consistency.
Validate query relevance.
