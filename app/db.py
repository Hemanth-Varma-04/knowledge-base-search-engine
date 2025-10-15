import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "kb_rag")
COL_NAME = os.getenv("MONGODB_COL", "chunks")

_client = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client

def get_collection():
    client = get_client()
    return client[DB_NAME][COL_NAME]
