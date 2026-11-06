import os
import hashlib
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance,VectorParams,PointStruct)

load_dotenv(override=True)

_SHARED_LOCAL_CLIENT = None

def get_shared_local_client(storage_path="data/qdrant_db"):
    global _SHARED_LOCAL_CLIENT
    if _SHARED_LOCAL_CLIENT is None:
        os.makedirs(storage_path, exist_ok=True)
        try:
            _SHARED_LOCAL_CLIENT = QdrantClient(path=storage_path)
        except RuntimeError as e:
            print(f"Notice: Storage folder '{storage_path}' is locked by another instance ({e}). Falling back to in-memory Qdrant storage.")
            _SHARED_LOCAL_CLIENT = QdrantClient(":memory:")
    return _SHARED_LOCAL_CLIENT

class QdrantStore:
    def __init__(self, collection_name="codebase", client=None):
        load_dotenv(override=True)
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

        # Check if local embedded storage or placeholder
        is_placeholder = "your-cluster-url" in qdrant_url or "your-qdrant-api-key" in qdrant_api_key
        self._is_local = qdrant_url.lower() in ("local", ":memory:", "") or is_placeholder

        if client is not None:
            self.client = client
        elif self._is_local:
            self._init_local_client()
        else:
            kwargs = {"url": qdrant_url}
            if qdrant_api_key and not is_placeholder:
                kwargs["api_key"] = qdrant_api_key
            try:
                self.client = QdrantClient(**kwargs)
            except Exception:
                self._switch_to_local()

        self.collection_name = collection_name

    def _init_local_client(self):
        self.client = get_shared_local_client("data/qdrant_db")
        self._is_local = True

    def _switch_to_local(self):
        print("Falling back to local embedded Qdrant storage at 'data/qdrant_db'")
        self._init_local_client()

    def list_collections(self):
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            if not self._is_local:
                self._switch_to_local()
                return self.list_collections()
            return []

    def create_collection(self, vector_size: int, recreate: bool = False):
        try:
            existing_collections = self.list_collections()
        except Exception:
            if not self._is_local:
                self._switch_to_local()
                existing_collections = self.list_collections()
            else:
                existing_collections = []

        if recreate and self.collection_name in existing_collections:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted existing collection for recreation: {self.collection_name}")
            existing_collections.remove(self.collection_name)

        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def add_chunks(self, chunks, embeddings, batch_size=100):
        total_chunks = len(chunks)
        if total_chunks == 0:
            return

        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            points = []

            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                point_id = int(hashlib.md5(chunk["chunk_id"].encode("utf-8")).hexdigest()[:16], 16)

                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        **chunk["metadata"]
                    }
                )

                points.append(point)

            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Uploaded chunks {start + 1}-{end} of {total_chunks}")

        print(f"Successfully uploaded {total_chunks} chunks")

    def search(self, query_vector, limit=5):
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=limit,
                with_payload=True,
            )
            return results.points
        except Exception as e:
            print(f"Error querying collection '{self.collection_name}': {e}")
            return []