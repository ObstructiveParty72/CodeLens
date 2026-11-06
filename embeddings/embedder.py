import functools
from sentence_transformers import SentenceTransformer


# Module-level singleton — loaded once per Python process
@functools.lru_cache(maxsize=1)
def _load_model(model_name: str):
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return SentenceTransformer(model_name, local_files_only=True)


class CodeEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = _load_model(model_name)

    def embed(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings