# RAG from Scratch — Local Retrieval-Augmented Generation

Built with LangChain, Sentence-Transformers, FAISS, Ollama (local LLM).
No OpenAI API. No cloud dependency.

## What it does
- Loads `.txt` and `.pdf` from `data/`
- Splits into chunks (300 chars, overlap 50)
- Embeds with `all-MiniLM-L6-v2` (384-dim)
- Stores in local FAISS index (`vector_store/`)
- Answers via semantic search + strict LLM rule + distance guard (`THRESHOLD = 1.4`)
- Refuses out-of-scope questions (e.g., "cricket match")
- Interfaces: Streamlit (`app.py`) + FastAPI (`api.py`)

## Run
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Put your resume/pdf in data/, then:
python src/build_vectorstore_v2.py
streamlit run app.py