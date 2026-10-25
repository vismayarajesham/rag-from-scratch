"""
STEP 8 — QUERY PHASE (retrieval only, no LLM yet)
Loads the SAVED index from disk. Does NOT rebuild anything.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    """Must be the SAME model used at build time."""
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# --- Load the index from disk (fast — no re-embedding!) ---
print("Loading saved index from disk...")
db = FAISS.load_local(
    "vector_store",
    LocalEmbeddings(),
    allow_dangerous_deserialization=True,   # explained below
)
print("Vectors in index:", db.index.ntotal)
print("Ready. Type your questions.\n")

# --- Interactive loop: now YOU are the user ---
while True:
    question = input("Ask a question (or 'quit'): ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    results = db.similarity_search(question, k=2)

    print(f"\nTop {len(results)} chunks the system would give the LLM:")
    for i, doc in enumerate(results):
        print(f"\n--- chunk {i} | source: {doc.metadata['source']} ---")
        print(doc.page_content[:200].replace("\n", " "))
    print()
