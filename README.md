# 📚 PDF RAG Question Answering System

A local Retrieval-Augmented Generation (RAG) system that allows users
to ask questions about a PDF book and receive answers based on the
retrieved content.

## 🚀 Project Overview

This project demonstrates how to build a RAG pipeline using:

- Python
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Ollama
- Llama

The system uses a PDF book as its knowledge source.

## 🔄 Architecture

PDF
↓
PDF Loader
↓
Text Chunking
↓
Metadata
↓
HuggingFace Embeddings
↓
ChromaDB
↓
User Question
↓
Similarity Search
↓
Relevant Chunks
↓
Prompt
↓
Ollama LLM
↓
Answer + Source Pages

## 🛠️ Technologies

- Python
- LangChain
- ChromaDB
- HuggingFace
- Sentence Transformers
- Ollama

## ⚙️ Configuration

The project uses `config.json` to configure:

- PDF path
- ChromaDB path
- Collection name
- Embedding model
- Chunk size
- Chunk overlap
- Top-K
- LLM model
- Temperature

## ▶️ How to Run

### 1. Clone the repository

git clone YOUR_REPOSITORY_URL

### 2. Create virtual environment

python -m venv .venv

### 3. Activate environment

Windows:

.venv\Scripts\activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Make sure Ollama is installed

Install and run the required model.

### 6. Update config.json

Set the correct PDF and database paths.

### 7. Run ingestion

python Ingestion.py

### 8. Ask questions

python QA.py

## 🔍 Example Questions

- Who is Ivan Ilyich?
- What was Ivan Ilyich's profession?
- Why did Ivan Ilyich become dissatisfied with his family?
- What role did Gerasim play during Ivan Ilyich's illness?
- What realization did Ivan Ilyich have near the end of his life?

## 🎯 Key Concepts Demonstrated

- Document loading
- Text chunking
- Metadata management
- Text embeddings
- Vector databases
- Semantic search
- Retrieval
- Prompt engineering
- Local LLM inference
- RAG architecture

## 📌 Future Improvements

- FastAPI backend
- Streamlit interface
- Hybrid search
- Reranking
- RAG evaluation
- Conversation memory
- Multiple PDF support