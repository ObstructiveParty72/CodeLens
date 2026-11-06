import re
from embeddings.embedder import CodeEmbedder
from vector_store.qdrant_store import QdrantStore
from ingestion.pipeline import IngestionPipeline


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name to comply with Qdrant collection naming rules
    (lowercase alphanumeric, hyphens, and underscores).
    """
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name).strip("_").lower()
    return clean or "codebase"


class RepositoryIndexer:
    def __init__(self, collection_name: str = "codebase", client=None):
        self.collection_name = sanitize_collection_name(collection_name)
        self.pipeline = IngestionPipeline()
        self.embedder = CodeEmbedder()
        self.store = QdrantStore(collection_name=self.collection_name, client=client)

    def index_repository(self, repo_url_or_path: str, recreate_collection: bool = False):
        result = self.pipeline.ingest(repo_url_or_path)
        chunks = result["chunks"]

        if not chunks:
            return {
                "repository_path": result["repository_path"],
                "files": len(result["files"]),
                "chunks": 0,
                "collection_name": self.collection_name,
            }

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        self.store.create_collection(
            vector_size=len(embeddings[0]),
            recreate=recreate_collection
        )
        self.store.add_chunks(chunks, embeddings)

        return {
            "repository_path": result["repository_path"],
            "files": len(result["files"]),
            "chunks": len(chunks),
            "collection_name": self.collection_name,
        }