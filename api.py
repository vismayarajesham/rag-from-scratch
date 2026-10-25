from fastapi import FastAPI
from pydantic import BaseModel
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

print("Loading index and model...")
db = FAISS.load_local("vector_store", LocalEmbeddings(), allow_dangerous_deserialization=True)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

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
THRESHOLD = 1.4

app = FastAPI(title="RAG from Scratch")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    refused_by: str | None

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    results = db.similarity_search_with_score(req.question, k=2)
    best_distance = results[0][1]
    if best_distance > THRESHOLD:
        return AskResponse(
            answer="I don't know - this information is not in my documents.",
            sources=[],
            refused_by="threshold",
        )
    context = "\n\n".join(
        f"[chunk {i}]\n{doc.page_content}" for i, (doc, dist) in enumerate(results)
    )
    prompt = (
        f"{SYSTEM_RULE}\n\nCONTEXT:\n{context}\n\n"
        f"QUESTION: {req.question}\n\nANSWER:"
    )
    answer = llm.invoke(prompt).content
    sources = sorted(
        {f"{doc.metadata['source']} (char {doc.metadata.get('start_index')})"
         for doc, dist in results}
    )
    return AskResponse(answer=answer, sources=sources, refused_by=None)