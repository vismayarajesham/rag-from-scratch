"""
STEP 7 — INDEXING PHASE
documents -> chunks -> vectors -> FAISS -> saved to disk
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    """Bridge between our local model and LangChain.

    LangChain doesn't care WHICH embedding model we use.
    It only demands two things (remember your doubt?):

      embed_documents() -> embed many chunks     (build time)
      embed_query()     -> embed one question    (query time)
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# 1. Load (same as before)
loader = TextLoader("data/novabot.txt", encoding="utf-8")
docs = loader.load()

# 2. Split (same as before)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, add_start_index=True
)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} doc(s) -> {len(chunks)} chunks")

# 3. Embed + store in FAISS — this ONE line replaces the manual
#    encode + cos_sim we wrote in Step 6
embeddings = LocalEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
print("Vectors inside the index:", db.index.ntotal)

# 4. SAVE to disk — this is the new superpower
db.save_local("vector_store")
print("Saved to vector_store/")

# 5. Quick test query
question = "How long does the battery last on one charge?"
results = db.similarity_search(question, k=2)

print("=" * 60)
print("Question:", question)
for i, doc in enumerate(results):
    print(f"\n--- result {i} ---")
    print(doc.page_content[:120].replace("\n", " "))
    print("metadata:", doc.metadata)