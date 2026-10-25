"""
STEP 10 — COMPLETE RAG
retrieve chunks -> build prompt -> local LLM -> grounded answer
"""

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    """SAME model as build time — the golden rule."""
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# --- 1. Load the saved index (no rebuilding) ---
print("Loading index from disk...")
db = FAISS.load_local(
    "vector_store",
    LocalEmbeddings(),
    allow_dangerous_deserialization=True,
)

# --- 2. The LLM (runs inside Ollama on YOUR machine) ---
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,     # 0 = factual & consistent, no creativity
)

# --- 3. THE RULE (the anti-hallucination shield) ---
SYSTEM_RULE = (
    "You are a support assistant for Nova Robotics.\n"
    "Answer the question using ONLY the context provided below.\n"
    "If the answer is not in the context, reply exactly:\n"
    "\"I don't know - this information is not in my documents.\"\n"
    "Do not use outside knowledge. Do not guess."
)

print("Ready. Ask your questions.\n")

# --- 4. The RAG loop ---
while True:
    question = input("Ask a question (or 'quit'): ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    # RETRIEVE
    results = db.similarity_search(question, k=2)

    # BUILD PROMPT  (this text is literally ALL the LLM receives)
    context = "\n\n".join(
        f"[chunk {i}]\n{doc.page_content}" for i, doc in enumerate(results)
    )
    prompt = (
        f"{SYSTEM_RULE}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    print("\n----- what the LLM actually receives -----")
    print(prompt)
    print("-------------------------------------------\n")

    # GENERATE
    print(end=" ", flush=True)
    answer = llm.invoke(prompt).content
    print(answer)
    print()