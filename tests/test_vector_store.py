from embeddings.embedder import CodeEmbedder
from vector_store.qdrant_store import QdrantStore
from ingestion.pipeline import IngestionPipeline

# ==================================================
# 1. Sample code chunks
# ==================================================
print("=" * 70)
print("INGESTING REPOSITORY")
print("=" * 70)

pipeline = IngestionPipeline()
repo_url = "https://github.com/psf/requests.git"
result = pipeline.ingest(repo_url)
chunks = result["chunks"]

print("Files:",len(result["files"]))
print("Chunks:",len(chunks))
# ==================================================
# 2. Create embedding model
# ==================================================
print("Loading embedding model...")
embedder = CodeEmbedder()

# ==================================================
# 3. Generate embeddings for code chunks
# ==================================================
texts = [
    chunk["content"]
    for chunk in chunks
]

embeddings = embedder.embed(texts)


print("Embedding dimension:",len(embeddings[0]))

# ==================================================
# 4. Connect to Qdrant Cloud
# ==================================================
print("Connecting to Qdrant Cloud...")

store = QdrantStore(collection_name="requests_codebase")


# ==================================================
# 5. Create collection
# ==================================================

store.create_collection(vector_size=len(embeddings[0]))


# ==================================================
# 6. Store chunks
# ==================================================

store.add_chunks(chunks,embeddings)

# ==================================================
# 7. Create query
# ==================================================
query = (
    "Where is user authentication "
    "implemented?"
)

# ==================================================
# 8. Embed query
# ==================================================

query_vector = embedder.embed([query])[0]

# ==================================================
# 9. Search Qdrant
# ==================================================

results = store.search(query_vector=query_vector,limit=3)

# ==================================================
# 10. Display results
# ==================================================
print()
print("=" * 70)
print("SEARCH RESULTS")
print("=" * 70)


for result in results:
    print()
    print("Score:",result.score)
    print("Chunk ID:",result.payload["chunk_id"])
    print("File:",result.payload["file_path"])
    print("Name:",result.payload["name"])
    print("Type:",result.payload["type"])
    print("Lines:",result.payload["start_line"],"-",result.payload["end_line"])
    print()
    print(result.payload["content"])
    print("-" * 70)