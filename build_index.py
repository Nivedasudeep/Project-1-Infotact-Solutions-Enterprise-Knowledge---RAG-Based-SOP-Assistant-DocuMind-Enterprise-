from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vectors
vectors = embeddings.embed_documents(chunks)

metadata = [{"source": "Employee-Handbook.pdf", "chunk_id": i} for i in range(len(chunks))]


#  FIX: convert to NumPy float32 array
vectors = np.array(vectors).astype("float32")
import json

metadata = [
    {"source": "Employee-Handbook.pdf", "chunk_id": i}
    for i in range(len(chunks))
]

with open("metadata.json", "w") as f:
    json.dump(metadata, f)

# Create FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# Save index
faiss.write_index(index, "faiss.index")

print("FAISS index created successfully")
