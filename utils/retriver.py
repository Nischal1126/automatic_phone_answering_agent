"""
debug_retrieval.py
Inspect exactly what the retriever returns for a query and why -
run this BEFORE assuming the LLM or vector DB is broken.
"""

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DB_DIR = r"C:\Users\nisch\OneDrive\Desktop\automated voice agent\data_prep\chroma_db"
COLLECTION_NAME = "doai_site"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Confirm the collection actually has data
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' has {collection.count()} chunks total.\n")

# 2. Run the real retrieval step with relevance scores visible
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
)

query = "what is the fee structure of Undergraduate"
results = vectordb.similarity_search_with_relevance_scores(query, k=6)

print(f'Top matches for: "{query}"\n')
for doc, score in results:
    print(f"score={score:.3f} | file={doc.metadata.get('source_file')} | section={doc.metadata.get('section')}")
    print(f"  text: {doc.page_content[:120]}...\n")