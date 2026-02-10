DocuMind Enterprise RAG

DocuMind Enterprise is a Retrieval-Augmented Generation (RAG) system built for enterprise use cases.
It ingests internal documents, retrieves relevant context using semantic search, and generates strictly grounded answers without hallucination.

This project was developed as part of Infotact Solutions – AI Research & Development Wing under the Enterprise Knowledge – RAG-Based SOP Assistant assignment.

🚀 Features

PDF document ingestion (enterprise policy documents)

Recursive text chunking with overlap

Semantic search using FAISS

Local embeddings using SentenceTransformers

Local LLM inference using HuggingFace (no paid APIs)

Hallucination guardrails (context-only answers)

FastAPI backend with streaming responses

Dockerized deployment

Health check endpoint

🛠 Tech Stack

Backend: FastAPI

Vector Store: FAISS

Embeddings: all-MiniLM-L6-v2

LLM: FLAN-T5 (HuggingFace)

PDF Parsing: Unstructured

Containerization: Docker

Language: Python 3.10
