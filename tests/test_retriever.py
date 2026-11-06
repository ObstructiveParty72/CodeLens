from retrieval.retriever import CodeRetriever
retriever = CodeRetriever()

question = "Where is user authentication implemented?"

results = retriever.retrieve(query=question,top_k=3)

print("=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)


for result in results:
    print()
    print("Score:", result.score)
    print("File:",result.payload.get("file_path"))
    print("Name:",result.payload.get("name"))
    print("Type:",result.payload.get("type"))
    print("Lines:",result.payload.get("start_line"),"-",result.payload.get("end_line"))
    print()
    print(result.payload.get("content"))
    print("-" * 70)