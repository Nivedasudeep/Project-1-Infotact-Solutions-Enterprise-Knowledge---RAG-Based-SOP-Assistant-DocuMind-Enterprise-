from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Tuple, List, Dict
import faiss
import numpy as np

app = FastAPI(title="DocuMind Enterprise RAG")

# -----------------------------
# Load embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index("faiss.index")

# -----------------------------
# Load documents
# -----------------------------
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# -----------------------------
# Load local LLM (CPU-safe)
# -----------------------------
llm = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_new_tokens=200,
    device=-1
)

# -----------------------------
# Retrieval function (WITH citations)
# -----------------------------
def retrieve_context(query: str, k: int = 3) -> Tuple[str, List[Dict]]:
    query_embedding = embedding_model.encode([query])
    D, I = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    retrieved_docs = []
    citations = []

    for i in I[0]:
        retrieved_docs.append(documents[i])
        citations.append({
            "source": "Employee-Handbook.pdf",
            "chunk_id": int(i)
        })

    return "\n".join(retrieved_docs), citations

# -----------------------------
# Streaming generator
# -----------------------------
def stream_answer(query: str):
    context, citations = retrieve_context(query)

    prompt = f"""
SYSTEM RULES (STRICT):
- You MUST answer ONLY using the provided context.
- If the answer is NOT explicitly present in the context, reply EXACTLY with:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    result = llm(prompt)[0]["generated_text"]

    # Stream answer
    for line in result.split("\n"):
        yield line + "\n"

    # Stream citations
    yield "\nSOURCES:\n"
    for c in citations:
        yield f"- {c['source']} (chunk {c['chunk_id']})\n"

# -----------------------------
# RAG API endpoint
# -----------------------------
@app.get("/rag/ask")
def rag_ask(query: str = Query(..., description="User question")):
    return StreamingResponse(
        stream_answer(query),
        media_type="text/plain"
    )

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
