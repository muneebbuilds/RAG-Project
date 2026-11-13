import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

CONFIG_FILE = "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    config = json.load(file)

DOCUMENT_PATH = config["document_path"]
DATABASE_PATH = config["database_path"]
METADATA_FILE = config["metadata_file"]
EMBEDDINGS_FILE = config["embeddings_file"]

COLLECTION_NAME = config["collection_name"]
EMBEDDING_MODEL = config["embedding_model"]

CHUNK_METHOD = config["chunk_method"]
CHUNK_SIZE = config["chunk_size"]
CHUNK_OVERLAP = config["chunk_overlap"]


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(DATABASE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)


# ============================================================
# 3. STARTUP INFORMATION
# ============================================================

print("=" * 60)
print("PDF RAG INGESTION")
print("=" * 60)

print(f"Document: {DOCUMENT_PATH}")
print(f"Database: {DATABASE_PATH}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Embedding: {EMBEDDING_MODEL}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunk overlap: {CHUNK_OVERLAP}")


# ============================================================
# 4. LOAD PDF
# ============================================================

print("\n[1/5] Loading PDF...")

loader = PyPDFLoader(DOCUMENT_PATH)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# ============================================================
# 5. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

print("\n[2/5] Creating chunks...")

if CHUNK_METHOD == "section":

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

else:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# ============================================================
# 6. ADD METADATA AND E5 DOCUMENT PREFIX
# ============================================================

print("\n[3/5] Adding metadata and prefixes...")

for chunk_id, chunk in enumerate(chunks):

    page = chunk.metadata.get("page")

    if page is not None:
        page += 1

    chunk.metadata.update({
        "chunk_id": chunk_id,
        "source": DOCUMENT_PATH,
        "page": page,
        "collection": COLLECTION_NAME
    })

    chunk.page_content = "passage: " + chunk.page_content

print("Metadata and prefixes added.")


# ============================================================
# 7. LOAD EMBEDDING MODEL
# ============================================================

print("\n[4/5] Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={"prompt": "query: "}
)

print("Embedding model loaded.")


# ============================================================
# 8. RESET AND STORE IN CHROMADB
# ============================================================

print("\n[5/5] Storing chunks in ChromaDB...")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DATABASE_PATH
)

# Remove old records before rebuilding
vectorstore.reset_collection()

# Insert new chunks
vectorstore.add_documents(chunks)

print("Chunks stored successfully.")


# ============================================================
# 9. SAVE METADATA
# ============================================================

print("\nSaving metadata...")

metadata = []

for chunk in chunks:

    metadata.append({
        "chunk_id": chunk.metadata.get("chunk_id"),
        "source": chunk.metadata.get("source"),
        "page": chunk.metadata.get("page"),
        "collection": chunk.metadata.get("collection"),
        "text": chunk.page_content
    })

with open(METADATA_FILE, "w", encoding="utf-8") as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )

print(f"Metadata saved: {METADATA_FILE}")
print(f"Metadata chunks: {len(metadata)}")


# ============================================================
# 10. SAVE EMBEDDINGS
# ============================================================

print("\nSaving embeddings...")

data = vectorstore.get(
    include=["embeddings"]
)

embeddings_data = data["embeddings"]

with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as file:

    for chunk_id, vector in enumerate(embeddings_data):

        file.write(f"Chunk {chunk_id + 1}\n")

        for value in vector:
            file.write(f"{value}\n")

        file.write("\n")

print(f"Embeddings saved: {EMBEDDINGS_FILE}")
print(f"Vectors saved: {len(embeddings_data)}")

if len(embeddings_data) > 0:
    print(f"Vector dimensions: {len(embeddings_data[0])}")


# ============================================================
# 11. VERIFY DATABASE AND METADATA
# ============================================================

print("\nVerifying database...")

total_records = vectorstore._collection.count()

print(f"ChromaDB records: {total_records}")
print(f"Expected chunks: {len(chunks)}")

if total_records == len(chunks):
    print("Database verification successful.")
else:
    print("WARNING: Database record count does not match.")


with open(METADATA_FILE, "r", encoding="utf-8") as file:

    saved_metadata = json.load(file)

print(f"Verified metadata chunks: {len(saved_metadata)}")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("INGESTION COMPLETE")
print("=" * 60)

print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")
print(f"Vectors: {len(embeddings_data)}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Database: {DATABASE_PATH}")
print(f"Metadata: {METADATA_FILE}")

print("\nPDF is ready for question answering!")
