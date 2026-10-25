from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load (same as before)
loader = TextLoader("data/novabot.txt", encoding="utf-8")
docs = loader.load()

# Create the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    add_start_index=True,
)

# Split
chunks = splitter.split_documents(docs)

print(f"Original docs: {len(docs)}")
print(f"Chunks after split: {len(chunks)}")
print("=" * 70)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} | length={len(chunk.page_content)} ---")
    print(chunk.page_content)
    print(f"metadata: {chunk.metadata}")