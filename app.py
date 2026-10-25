import streamlit as st
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

@st.cache_resource
def load():
    db = FAISS.load_local("vector_store", LocalEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    return db, llm

db, llm = load()

SYSTEM = ("Answer ONLY from the provided document chunks. "
          "If not found, say: I don't know - this is not in my documents. "
          "Never guess. Always cite sources.")
THRESHOLD = 1.4

st.title("RAG Demo - NovaBot Documents")
st.markdown("Semantic search + Local LLM (Ollama). Built with LangChain + FAISS.")

q = st.text_input("Ask a question:", "What is the warranty of NovaBot X2?")
if st.button("Ask"):
    res = db.similarity_search_with_score(q, k=2)
    best = res[0][1]
    if best > THRESHOLD:
        st.write("I don't know - this is not in my documents.")
        st.write(f"(threshold guard, distance {best:.2f})")
    else:
        ctx = "\n\n".join(f"[chunk {i}]\n{d.page_content}" for i,(d,_) in enumerate(res))
        prompt = f"{SYSTEM}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {q}\n\nANSWER:"
        ans = llm.invoke(prompt).content
        st.subheader("Answer")
        st.write(ans)
        with st.expander("Sources / Retrieved chunks"):
            for d,_ in res:
                st.write(f"→ {d.metadata['source']} (char {d.metadata.get('start_index')})")