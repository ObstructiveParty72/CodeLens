import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingestion.pipeline import IngestionPipeline
from ingestion.indexer import RepositoryIndexer, sanitize_collection_name
from retrieval.rag_pipeline import RAGPipeline
from vector_store.qdrant_store import QdrantStore

load_dotenv()

app = FastAPI(
    title="Codebase RAG",
    description="AI Software Engineer for understanding software repositories",
    version="0.2.0"
)


class RepositoryAnalyzeRequest(BaseModel):
    repo_url_or_path: str = Field(..., description="GitHub URL or local filesystem path")


class RepositoryIndexRequest(BaseModel):
    repo_url_or_path: str = Field(..., description="GitHub URL or local filesystem path")
    collection_name: str | None = Field(default=None, description="Optional custom Qdrant collection name")
    recreate: bool = Field(default=False, description="Recreate collection if it already exists")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question about the codebase")
    collection_name: str = Field(default="codebase", description="Target collection name")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    provider: str | None = Field(default=None, description="LLM provider: 'gemini' or 'groq'")
    model: str | None = Field(default=None, description="Model identifier (e.g. 'gemini-2.5-flash')")


@app.get("/")
def root():
    return {"message": "Welcome to Codebase RAG API!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/collections")
def list_collections():
    try:
        store = QdrantStore()
        return {"collections": store.list_collections()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repository/analyze")
def analyze_repository(request: RepositoryAnalyzeRequest):
    try:
        pipeline = IngestionPipeline()
        result = pipeline.ingest(request.repo_url_or_path)
        return {
            "repository_path": result["repository_path"],
            "total_files": len(result["files"]),
            "total_chunks": len(result["chunks"]),
            "files": result["files"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repository/index")
def index_repository(request: RepositoryIndexRequest):
    try:
        # Determine collection name if not specified
        if not request.collection_name:
            target_name = Path(request.repo_url_or_path.rstrip("/")).stem
            collection_name = sanitize_collection_name(f"{target_name}_codebase")
        else:
            collection_name = sanitize_collection_name(request.collection_name)

        indexer = RepositoryIndexer(collection_name=collection_name)
        result = indexer.index_repository(
            repo_url_or_path=request.repo_url_or_path,
            recreate_collection=request.recreate
        )
        return {
            "status": "success",
            "collection_name": collection_name,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_codebase(request: QueryRequest):
    try:
        collection = sanitize_collection_name(request.collection_name)
        rag = RAGPipeline(
            collection_name=collection,
            model=request.model,
            provider=request.provider
        )
        result = rag.answer(question=request.question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
