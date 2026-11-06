from retrieval.retriever import CodeRetriever
from retrieval.context_builder import ContextBuilder
from llm.client import LLMClient


class RAGPipeline:
    def __init__(
        self,
        collection_name: str = "codebase",
        model: str | None = None,
        provider: str | None = None,
        client=None
    ):
        self.collection_name = collection_name
        self.retriever = CodeRetriever(collection_name=collection_name, client=client)
        self.context_builder = ContextBuilder()
        self.llm = LLMClient(provider=provider, model=model)

    def answer(self, question: str, top_k: int = 5):
        # 1. Retrieve relevant code
        results = self.retriever.retrieve(query=question, top_k=top_k)

        if not results:
            return {
                "question": question,
                "answer": (
                    "No relevant code snippets were found in the codebase. "
                    "Please ensure the repository has been indexed properly."
                ),
                "sources": []
            }

        # 2. Build context
        context = self.context_builder.build(results)

        # 3. Generate answer
        answer = self.llm.generate(question=question, context=context)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "file_path": result.payload.get("file_path"),
                    "name": result.payload.get("name"),
                    "type": result.payload.get("type"),
                    "start_line": result.payload.get("start_line"),
                    "end_line": result.payload.get("end_line"),
                    "score": getattr(result, "score", 0.0),
                    "content": result.payload.get("content", ""),
                }
                for result in results
            ]
        }