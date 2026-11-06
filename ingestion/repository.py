from pathlib import Path
from .file_filter import get_language, should_ignore
from .parser import CodeParser
from .extractor import CodeExtractor

class RepositoryScanner:
    def __init__(self):
        self.parser = CodeParser()
        self.extractor = CodeExtractor()

    def scan(self, repository_path: str):
        repository_path = Path(repository_path)
        if not repository_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repository_path}")

        files = []
        for file_path in repository_path.rglob("*"):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(repository_path)
            if should_ignore(relative_path):
                continue
            language = get_language(file_path)
            if language is None:
                continue

            try:
                source_code = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            parsed_code = None

            if self.parser.is_supported(language):
                try:
                    tree = self.parser.parse(source_code, language)
                    parsed_code = self.extractor.extract(tree, source_code)
                except Exception:
                    parsed_code = None

            files.append({
                "path": relative_path.as_posix(),
                "language": language,
                "extension": file_path.suffix.lower(),
                "size": file_path.stat().st_size,
                "code": parsed_code,
                "content": source_code,
            })
        return files
