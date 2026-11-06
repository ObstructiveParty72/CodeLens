from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

repo_url = "https://github.com/psf/requests.git"

result = pipeline.ingest(repo_url)

print("=" * 70)
print("REPOSITORY:")
print(result["repository_path"])
print("=" * 70)
print("TOTAL FILES:")
print(len(result["files"]))
print("=" * 70)
print("TOTAL CHUNKS:")
print(len(result["chunks"]))
print("=" * 70)
print("FIRST 5 CHUNKS:")

for chunk in result["chunks"][:5]:
    print()
    print("Chunk ID:", chunk["chunk_id"])
    print("File:",chunk["metadata"]["file_path"])
    print("Type:",chunk["metadata"]["type"])
    print("Name:",chunk["metadata"]["name"])
    print("Lines:",chunk["metadata"]["start_line"],"-",chunk["metadata"]["end_line"])
    print()
    print(chunk["content"])
    print("-" * 70)