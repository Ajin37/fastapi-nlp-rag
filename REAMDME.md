# FastAPI NLP Application

## Overview

This project is a FastAPI-based NLP microservice that supports:

- Text Summarization  
- Topic Classification  
- Named Entity Extraction  
- Sentiment Analysis  

Each task is powered by an external LLM (UltraSafe) and enhanced using Retrieval-Augmented Generation (RAG) with vector store support and reranking.

---

## Installation Instructions

### Prerequisites

- Python 3.9+
- `virtualenv` or `venv`
- `pip`

### Setup

```bash
git clone https://github.com/Ajin37/fastapi-nlp-rag
cd FASTAPI-NLP
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the App

```bash
uvicorn main:app --reload
```
The API will be available at: http://127.0.0.1:8000
Swagger UI docs: http://127.0.0.1:8000/docs

### Implementation Details
- Language Model: All NLP tasks use the UltraSafe LLM via chat-style prompts.
- Vector Store: In-memory persistence using custom vector store saved to disk (vector_store_data/).
- Embedding + RAG: We retrieve top-k similar samples using embeddings and rerank them for better context.
- Modular Services: Each NLP task (summarization, classification, etc.) is implemented in its own service layer.
- Async I/O: All external API calls are async using httpx.

### Design Decisions & Trade-offs
- RAG + Reranker: Improves performance for NLP tasks by using semantically similar prior examples. This is beneficial especially when model context is limited or few-shot learning helps.
- Simple Vector Store: Chose an in-memory vector store with file persistence to reduce external dependencies (e.g., no Redis or any other cloud vdb for simplicity).
- Modular Structure: Split routes, services, and utilities to support easier scaling, debugging, and future extensions.
- No ORM or DB: Vector store was sufficient for current tasks; DB integration could be added if needed for user/task tracking.