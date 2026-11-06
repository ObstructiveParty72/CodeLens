from pathlib import Path
from ingestion.github_loader import GitHubLoader
from ingestion.repository import RepositoryScanner
from ingestion.chunker import CodeChunker


class IngestionPipeline:
    def __init__(self):
        self.loader = GitHubLoader()
        self.scanner = RepositoryScanner()
        self.chunker = CodeChunker()

    def ingest(self, repo_url_or_path: str):
        # -----------------------------------------
        # 1. Resolve local path or clone Git repository
        # -----------------------------------------
        local_path = Path(repo_url_or_path)
        if local_path.is_dir():
            repository_path = str(local_path.resolve())
            print(f"Using local repository path: {repository_path}")
        else:
            repository_path = self.loader.clone_repository(repo_url_or_path)

        # -----------------------------------------
        # 2. Scan repository
        # -----------------------------------------
        files = self.scanner.scan(repository_path)

        # -----------------------------------------
        # 3. Create code chunks
        # -----------------------------------------
        chunks = self.chunker.create_chunks(files)

        return {
            "repository_path": repository_path,
            "files": files,
            "chunks": chunks,
        }