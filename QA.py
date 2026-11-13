import json
import requests

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# CONFIGURATION

CONFIG_FILE = "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    config = json.load(file)

DATABASE_PATH = config["database_path"]
COLLECTION_NAME = config["collection_name"]
EMBEDDING_MODEL = config["embedding_model"]
TOP_K = config["top_k"]

LLM_MODEL = config["model"]
TEMPERATURE = config["temperature"]

API_KEY = config["api_key"]
BASE_URL = config["base_url"].rstrip("/")


# STARTUP

print("=" * 60)
print("PDF RAG: QUESTION ANSWERING")
print("=" * 60)

print(f"Collection: {COLLECTION_NAME}")
print(f"LLM Model: {LLM_MODEL}")


# LOAD EMBEDDINGS

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding Model Loaded")


# LOAD CHROMADB

print("\nLoading ChromaDB...")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DATABASE_PATH
)

print("ChromaDB Loaded")

print("Total stored records:", vectorstore._collection.count())


# CREATE RETRIEVER

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)


# QUESTION ANSWERING FUNCTION

def ask_question(question):

    documents = retriever.invoke(question)

    if not documents:
        return "Not found in the retrieved passages.", []

    context_parts = []

    for document in documents:

        page = document.metadata.get("page")
        text = document.page_content

        context_parts.append(
            f"[Page {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)


    # PROMPT

    prompt = f"""
You answer questions about The Death of Ivan Ilyich.

Use ONLY the provided context.

Rules:
- Never use outside knowledge.
- Do not invent or assume facts.
- If the answer is unsupported, say:
  "Not found in the retrieved passages."
- Answer in 2-4 sentences.
- Mention only relevant source pages.

Context:
{context}

Question:
{question}

Answer:
"""


    # SEND REQUEST TO KAGGLE GPU

    response = requests.post(
        f"{BASE_URL}/api/generate",
        headers={
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "temperature": TEMPERATURE,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    answer = response.json()["response"]

    return answer, documents


# QUESTION ANSWERING LOOP

print("\nAsk questions about The Death of Ivan Ilyich.")
print("Type 'exit' to stop.")

while True:

    question = input("\nQuestion: ").strip()

    if question.lower() in ["exit", "quit", "q"]:
        print("\nGOODBYE!")
        break

    if not question:
        continue

    try:

        answer, source_documents = ask_question(question)

        print("\nAnswer:")
        print("-" * 60)
        print(answer)

        print("\nRetrieved Sources:")
        print("-" * 60)

        for document in source_documents:

            page = document.metadata.get("page")
            chunk_id = document.metadata.get("chunk_id")

            print(f"Page: {page} | Chunk ID: {chunk_id}")

        print("-" * 60)

    except requests.exceptions.Timeout:
        print("\nError: Kaggle GPU request timed out.")

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the Kaggle API.")

    except requests.exceptions.HTTPError as error:
        print(f"\nAPI Error: {error}")
        print("Response:", error.response.text)

    except (KeyError, ValueError) as error:
        print(f"\nInvalid API response: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {error}")