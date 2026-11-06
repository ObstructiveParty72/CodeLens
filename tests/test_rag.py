from retrieval.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

question = (
    "Where is user authentication "
    "implemented?"
)
result = pipeline.answer(question=question,top_k=5)

print("=" * 70)
print("ANSWER")
print("=" * 70)

print()

print(result["answer"])

print()
print("=" * 70)
print("SOURCES")
print("=" * 70)


for source in result["sources"]:
    print()
    print("File:",source["file_path"])
    print("Name:",source["name"])
    print("Type:",source["type"])
    print("Lines:",source["start_line"],"-",source["end_line"])
    print("Score:",source["score"])