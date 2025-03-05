import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List

# Load FAISS index and documents into memory (Avoid reloading on each query)
INDEX_PATH = "faiss_index.idx"
DOCS_PATH = "doc_texts.npy"

if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    print("Loading FAISS index and documents into memory...")
    index = faiss.read_index(INDEX_PATH)
    doc_texts = np.load(DOCS_PATH, allow_pickle=True)
else:
    raise FileNotFoundError("FAISS index or document data not found. Run indexing first.")

# Load optimized embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class UserQuery(BaseModel):
    query: str

class RetrievedSection(BaseModel):
    text: str

def query_legal_documents(query: UserQuery, top_k=3) -> List[RetrievedSection]:
    """Retrieve top-k relevant legal sections using FAISS."""
    query_embedding = embedding_model.encode([query.query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_sections = [RetrievedSection(text=doc_texts[i]) for i in indices[0] if i < len(doc_texts)]
    return retrieved_sections