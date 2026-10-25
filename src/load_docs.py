from langchain_community.document_loaders import TextLoader

# 1. Point the loader at our file
loader = TextLoader("data/novabot.txt", encoding="utf-8")

# 2. Load it
docs = loader.load()

# 3. Inspect what we got
print("Number of documents:", len(docs))
print("Python type:", type(docs[0]))
print()

print("--- page_content (first 300 chars) ---")
print(docs[0].page_content[:300])
print()

print("--- metadata ---")
print(docs[0].metadata)
print()

print("--- total characters ---")
print(len(docs[0].page_content))