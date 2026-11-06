from embeddings.embedder import CodeEmbedder

embedder = CodeEmbedder()

texts = [
    "def authenticate(token):",
    "connect to redis database",
    "calculate fibonacci number"
]

embeddings = embedder.embed(texts)

print("Number of embeddings:", len(embeddings))

print("Embedding dimension:",len(embeddings[0]))

print( "First embedding:", embeddings[0] )