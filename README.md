# PDF RAG Project

A Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents and receive context-based answers with relevant source references.

This project uses LangChain, ChromaDB, HuggingFace embeddings, and a remote Ollama LLM hosted on Kaggle GPU.

## Key Features

- **PDF Processing:** Extracts text from PDF documents.
- **Text Chunking:** Divides extracted text into smaller, manageable chunks.
- **Semantic Search:** Finds relevant document passages using vector embeddings.
- **Vector Database:** Stores and retrieves embeddings using ChromaDB.
- **Context-Aware Question Answering:** Generates answers using retrieved document context.
- **Source References:** Displays page numbers of retrieved passages.
- **Configurable Pipeline:** Allows customization of chunk size, overlap, Top-K retrieval, and model settings.
- **Remote LLM Inference:** Uses Qwen 2.5 7B through Ollama on Kaggle GPU.
- **API Integration:** Connects the local application to the remote LLM through FastAPI and Cloudflare Tunnel.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline development |
| ChromaDB | Vector database |
| HuggingFace | Embedding model integration |
| E5-small-v2 | Text embedding model |
| Ollama | LLM inference |
| Qwen 2.5 7B | Language model |
| FastAPI | API gateway |
| Cloudflare Tunnel | Remote API connectivity |

## Project Architecture

The system follows this workflow:

1. Load and extract text from a PDF document.
2. Split extracted text into smaller chunks.
3. Generate vector embeddings using E5-small-v2.
4. Store embeddings in ChromaDB.
5. Accept a question from the user.
6. Retrieve the most relevant document chunks through semantic search.
7. Send the question and retrieved context to the remote LLM.
8. Generate an answer based on the provided context.
9. Display the answer and retrieved source references.

## LLM Deployment Architecture

The project uses a hybrid local and cloud-based architecture.

**Local Machine**
- PDF processing
- Text chunking
- Embedding generation
- ChromaDB storage
- Semantic retrieval
- Question-answering logic

**Kaggle GPU Environment**
- Ollama
- Qwen 2.5 7B
- FastAPI gateway

**Cloudflare Tunnel**
- Provides connectivity between the local application and the Kaggle-hosted API.

This architecture allows the application to use a larger language model without requiring it to run locally.

## Configuration

Project settings are managed through `config.json`.

The configuration includes:

- Document path
- Database path
- Embedding model
- Collection name
- Chunk size
- Chunk overlap
- Top-K retrieval value
- LLM model
- Temperature
- API credentials and endpoint

**Security:** API credentials should never be committed to GitHub. Keep your actual `config.json` private and provide a sanitized `config.example.json` for reference.

## Project Structure

```text
RAG-Project/
│
├── Data/
│   └── The-Death-of-Ivan.pdf
│
├── Database/
│   ├── chroma_db/
│   ├── chunk_metadata.json
│   └── embeddings.txt
│
├── Ingestion.py
├── QA.py
├── config.example.json
├── .gitignore
└── README.md
```

*Generated database files and the original PDF may be excluded from the repository.*

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd RAG-Project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install langchain langchain-chroma langchain-huggingface chromadb sentence-transformers requests
```

### 4. Configure the Application

Create your local `config.json` using `config.example.json`.

Add your document paths, database settings, model configuration, API endpoint, and private API token.

## Running the Project

### Step 1: Ingest the PDF

```bash
python Ingestion.py
```

This extracts the document text, creates chunks, generates embeddings, and stores them in ChromaDB.

**Note:** Run ingestion again only when processing a new or updated document, or rebuilding the vector database.

### Step 2: Start the Remote LLM Environment

Start your Kaggle notebook and ensure Ollama, Qwen 2.5 7B, FastAPI, and Cloudflare Tunnel are running.

### Step 3: Start Question Answering

```bash
python QA.py
```

Enter your questions and receive answers based on the retrieved PDF passages.

Type `exit` to close the application.

## Current Configuration

| Parameter | Value |
|---|---|
| Embedding Model | intfloat/e5-small-v2 |
| Chunk Size | 1000 |
| Chunk Overlap | 200 |
| Top-K | 5 |
| LLM | Qwen 2.5 7B |
| LLM Runtime | Ollama |
| Vector Database | ChromaDB |

## Learning Outcomes

Through this project, I explored:

- Retrieval-Augmented Generation architecture.
- Document processing and text chunking.
- Embeddings and semantic search.
- Vector database implementation.
- LangChain integration.
- Context-based question answering.
- Local and remote API communication.
- GPU-based LLM inference.

## Future Improvements

- Improve document retrieval accuracy.
- Implement hybrid search.
- Experiment with reranking techniques.
- Add support for multiple PDF documents.
- Implement RAG evaluation metrics.
- Develop a web-based user interface.

---

**Author:** Muhammad Muneeb

**Project:** PDF Retrieval-Augmented Generation System