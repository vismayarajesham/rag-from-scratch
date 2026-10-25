"""
STEP 12 — INDEX EVERYTHING IN data/
.txt files  -> TextLoader   -> 1 Document per file
.pdf files  -> PyPDFLoader  -> 1 Document PER PAGE (metadata gets 'page')
Then: split -> embed -> FAISS -> save (overwrites the old vector_store)
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# --- 1. Load EVERY supported file in data/ ---
all_docs = []
for file in sorted(Path("data").iterdir()):
    suffix = file.suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(file), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file))
    else:
        print(f"skipped (unsupported): {file.name}")
        continue

    page_docs = loader.load()
    print(f"{file.name:30} -> {len(page_docs)} document(s)")
    all_docs.extend(page_docs)

print(f"\nTotal documents loaded: {len(all_docs)}")

# show what PDF metadata looks like
for d in all_docs:
    if d.metadata.get("source", "").endswith(".pdf"):
        print("\nExample PDF Document metadata:", d.metadata)
        break

# --- 2. Split (SAME as always) ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, add_start_index=True
)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks after split: {len(chunks)}")

# --- 3. Embed + store (SAME as always) ---
db = FAISS.from_documents(chunks, LocalEmbeddings())
print("Vectors in index:", db.index.ntotal)

# --- 4. Save (SAME as always) ---
db.save_local("vector_store")
print("Saved to vector_store/ — your knowledge base is now txt + pdf")