# RAG from Scratch — Local Retrieval-Augmented Generation

A fully local Retrieval-Augmented Generation (RAG) system built from scratch using LangChain, Sentence-Transformers, FAISS, and a local Ollama LLM. No OpenAI API keys. No cloud dependency.

The system indexes documents (`.txt` and `.pdf`), performs semantic search, retrieves relevant context chunks, applies a distance-threshold refusal guard, and generates grounded answers with source citations.

---

## Features

- Document Indexing (`.txt` + `.pdf`)
- Semantic Search with Embeddings (`all-MiniLM-L6-v2`)
- Chunk Splitting with Overlap (`RecursiveCharacterTextSplitter`)
- Vector Storage & Retrieval (FAISS)
- Local LLM Inference (`llama3.2:3b` via Ollama)
- Distance-Based Refusal Guard (`THRESHOLD = 1.4`)
- Grounded Answers with Source Citations
- FastAPI REST Endpoint (`/ask`)
- Streamlit Chat UI (`app.py`)
- Multi-Document Support (tested with `novabot.txt` + `novabot_x2_manual.pdf`)

---

## Tech Stack

- Python 3.12
- LangChain (loaders, splitters, embeddings, retrieval chains)
- LangChain-Ollama (`ChatOllama`)
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- FAISS (CPU)
- PyPDF / Pypdf (`PyPDFLoader`)
- Streamlit (`app.py`)
- FastAPI + Uvicorn (`api.py`)
- Ollama (local LLM runner)

---

## Project Structure
rag-from-scratch/
├── data/ # Documents to index (txt + pdf)
│ ├── novabot.txt
│ └── novabot_x2_manual.pdf
├── src/ # Step-by-step pipeline scripts
│ ├── load_docs.py
│ ├── split_docs.py
│ ├── embed_docs.py
│ ├── build_vectorstore.py
│ ├── build_vectorstore_v2.py
│ ├── ask.py
│ ├── rag.py
│ ├── rag_v2.py
│ ├── embed_docs.py
│ └── load_docs.py
├── api.py # FastAPI endpoint (POST /ask)
├── app.py # Streamlit UI interface
├── vector_store/ # FAISS index (generated)
│ ├── index.faiss
│ └── index.pkl
├── .env # (not committed)
├── .gitignore
└── requirements.txt

text


---

## How It Works

1. **Load** → `TextLoader` / `PyPDFLoader` reads files into `Document` objects
2. **Split** → chunks of ~300 chars with 50-char overlap
3. **Embed** → `SentenceTransformer` converts chunks to 384-dim vectors
4. **Index** → `FAISS.from_documents()` builds and saves the vector DB
5. **Retrieve** → user question is embedded; `similarity_search` returns top chunks
6. **Guard** → if best distance > 1.4, refuse instantly (no LLM call)
7. **Generate** → local LLM (`llama3.2:3b`) answers from retrieved context only
8. **Cite** → answer includes source file + start index from metadata

---

## Tested Demonstrations

| Question | Source | Result |
|---|---|---|
| "What is warranty of NovaBot X2?" | `novabot_x2_manual.pdf` | 24 months |
| "What is price of NovaBot X2?" | `novabot_x2_manual.pdf` | 49,999 rupees |
| "What is warranty of NovaBot X1?" | `novabot.txt` | 18 months |
| "Who won cricket match?" | — | Refused by threshold guard |
| "What is price of NovaBot X1?" | — | Refused (fact not in index) |

---

## Installation & Run

```bash
# 1. Clone and create environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# 2. Install
pip install -r requirements.txt

# 3. Pull local LLM (one-time, ~2 GB)
ollama pull llama3.2:3b

# 4. Build index (after placing your .txt / .pdf in data/)
python src/build_vectorstore_v2.py

# 5. Launch Streamlit UI
streamlit run app.py
# Open: http://localhost:8501

# 6. Launch FastAPI (optional, separate terminal)
uvicorn api:app --reload
# Open: http://127.0.0.1:8000/docs
Live Demo / Screenshot
Local Streamlit interface: http://localhost:8501

Demo Screenshot
<img width="1808" height="865" alt="Screenshot 2026-10-25 120927" src="https://github.com/user-attachments/assets/e64cc795-0786-4858-aeb5-ad3465a9c6c6" />

<img width="1897" height="899" alt="Screenshot 2026-10-25 120848" src="https://github.com/user-attachments/assets/dd55e571-3f1a-4d8e-8b5e-0469e20616ad" />

Author
Vismaya Rajesh A M B.Tech Computer Science and Engineering
