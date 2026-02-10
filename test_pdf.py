from unstructured.partition.pdf import partition_pdf
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = Path("data/docs/Employee-Handbook.pdf")

# Load PDF
elements = partition_pdf(filename=str(pdf_path))

# Combine all text
full_text = "\n".join([el.text for el in elements if el.text])

print(f"Total characters in document: {len(full_text)}")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(full_text)

print(f"Total chunks created: {len(chunks)}")
print("\nSample chunk:\n")
print(chunks[0])
