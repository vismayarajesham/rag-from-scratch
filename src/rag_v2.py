"""
STEP 11 — RAG v2
+ shows the distance score of every retrieved chunk
+ shows SOURCES (which file/position the answer came from)
+ hardened refusal prompt (few-shot example)
+ optional auto-refusal threshold (skips the LLM when nothing matches)
"""

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# --- setup ---
db = FAISS.load_local(
    "vector_store", LocalEmbeddings(), allow_dangerous_deserialization=True
)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# HARDENED RULE — now with a worked example ("few-shot").
# Small models copy examples much better than abstract instructions.
SYSTEM_RULE = (
    "You are a support assistant for Nova Robotics products.\n"
    "Rules:\n"
    "1. Answer ONLY using the CONTEXT given below.\n"
    "2. If the CONTEXT does not contain the answer, reply exactly:\n"
    "   I don't know - this information is not in my documents.\n"
    "3. Never use your own knowledge. Never guess numbers, dates or names.\n"
    "\nExample of correct behaviour:\n"
    "CONTEXT: The X200 charger outputs 24V.\n"
    "QUESTION: What is the capital of France?\n"
    "ANSWER: I don't know - this information is not in my documents."
)

# Auto-refusal guard: if the BEST chunk is still farther than this,
# skip the LLM entirely and refuse directly (fast + saves CPU).
# None = disabled for now — we will set the real number TOGETHER
# after you paste your observed scores.
THRESHOLD = 1.4

print("Ready. Ask your questions.\n")

while True:
    question = input("Ask a question (or 'quit'): ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    # RETRIEVE — now with scores
    results = db.similarity_search_with_score(question, k=2)

    print("\n--- retrieved chunks (distance: LOWER = more similar) ---")
    for i, (doc, dist) in enumerate(results):
        print(
            f"chunk {i} | distance {dist:.3f} | "
            f"{doc.metadata['source']} | starts at char {doc.metadata.get('start_index')}"
        )

    # GUARD (inactive until THRESHOLD gets a number)
    best_distance = results[0][1]
    if THRESHOLD is not None and best_distance > THRESHOLD:
        print("\nAnswer: I don't know - this information is not in my documents.")
        print("(refused by threshold guard — LLM was not even called)\n")
        continue

    # BUILD PROMPT
    context = "\n\n".join(
        f"[chunk {i}]\n{doc.page_content}" for i, (doc, dist) in enumerate(results)
    )
    prompt = (
        f"{SYSTEM_RULE}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    # GENERATE
    answer = llm.invoke(prompt).content
    print("\nAnswer:", answer)

    # CITATIONS — where the answer came from
    sources = sorted(
        {f"{doc.metadata['source']} (char {doc.metadata.get('start_index')})"
         for doc, dist in results}
    )
    print("Sources:", "; ".join(sources))
    print()