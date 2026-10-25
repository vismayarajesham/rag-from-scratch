from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

# --- Load + split (same as Step 5) ---
loader = TextLoader("data/novabot.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    add_start_index=True,
)
chunks = splitter.split_documents(docs)

# --- Load the embedding model (downloads on first run) ---
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Embed every chunk ---
texts = [c.page_content for c in chunks]
vectors = model.encode(texts)

print(f"Chunks: {len(chunks)}")
print(f"Vectors shape: {vectors.shape}")
print(f"First 5 numbers of chunk 0: {vectors[0][:5]}")
print("=" * 60)

# --- Semantic search demo ---
question = input("Ask a question: ")
print(f"Question: {question}\n")

q_vec = model.encode(question)
similarities = util.cos_sim(q_vec, vectors)[0]

# Sort chunks by similarity, highest first
ranked = sorted(
    range(len(chunks)),
    key=lambda i: similarities[i],
    reverse=True,
)

for rank, i in enumerate(ranked):
    score = similarities[i].item()
    print(f"rank {rank} | score {score:.4f} | chunk {i}")
    print(chunks[i].page_content[:100].replace("\n", " "))
    print()