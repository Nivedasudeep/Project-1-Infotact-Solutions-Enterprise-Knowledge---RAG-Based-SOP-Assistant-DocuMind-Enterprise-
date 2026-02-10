from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
pdf_path = Path("data/docs/Employee-Handbook.pdf")
elements = partition_pdf(filename=str(pdf_path))

# Combine extracted text
full_text = "\n".join([el.text for el in elements if el.text])

# Chunk text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_text(full_text)

print(f"Chunks created: {len(chunks)}")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS index
vector_store = FAISS.from_texts(
    chunks,
    embeddings,
    metadatas=[{"source": "Employee-Handbook.pdf", "chunk_id": i} for i, _ in enumerate(chunks)]
)
vector_store.save_local("vector_store")
print("Vector store with metadata saved successfully.")