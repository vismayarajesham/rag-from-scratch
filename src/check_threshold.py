"""
Measures the best FAISS distance for in-scope vs out-of-scope questions.
Retriever only — no LLM. We use these numbers to pick THRESHOLD.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


db = FAISS.load_local(
    "vector_store", LocalEmbeddings(), allow_dangerous_deserialization=True
)

test_questions = [
    # in-scope (answer exists in novabot.txt)
    "What is the warranty period of the NovaBot X1?",
    "How long does the battery last?",
    "How do I contact customer support?",
    "How often do firmware updates come?",
    # out-of-scope (no answer in the document)
    "Who won the cricket match yesterday?",
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
]

print(f"{'distance':>9}  {'scope':6}  question")
print("-" * 70)

for q in test_questions:
    results = db.similarity_search_with_score(q, k=1)
    dist = results[0][1]
    scope = "IN" if q in test_questions[:4] else "OUT"
    print(f"{dist:9.3f}  {scope:6}  {q}")