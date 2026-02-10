from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load embeddings (must match build step)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing FAISS index
vector_store = FAISS.load_local(
    "vector_store",
    embeddings
)


# Test query
query = "What is the company's policy on workplace harassment?"

docs = vector_store.similarity_search(query, k=3)

print("\nTop Retrieved Chunks:\n")
for i, doc in enumerate(docs, 1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content[:500])
    print()
