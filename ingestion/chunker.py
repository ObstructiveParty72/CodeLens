from pathlib import Path

class CodeChunker:
    def __init__(self, max_lines: int = 60, line_overlap: int = 10):
        self.max_lines = max_lines
        self.line_overlap = line_overlap

    def create_chunks(self, files):
        chunks = []

        for file in files:
            code_elements = file.get("code")

            if code_elements:
                for index, element in enumerate(code_elements):
                    chunk = {
                        "chunk_id": f"{file['path']}::{element['type']}::{index}",
                        "content": element["content"],
                        "metadata": {
                            "file_path": file["path"],
                            "language": file.get("language", "text"),
                            "type": element["type"],
                            "name": element["name"],
                            "start_line": element["start_line"],
                            "end_line": element["end_line"],
                        }
                    }
                    chunks.append(chunk)
            else:
                # Fallback to line-windowed chunking for text/code files without AST
                fallback_chunks = self._chunk_text(file)
                chunks.extend(fallback_chunks)

        return chunks

    def _chunk_text(self, file):
        content = file.get("content", "")
        if not content or not content.strip():
            return []

        lines = content.splitlines()
        total_lines = len(lines)
        file_name = Path(file["path"]).name
        language = file.get("language", "text")

        if total_lines <= self.max_lines:
            return [{
                "chunk_id": f"{file['path']}::file::0",
                "content": content,
                "metadata": {
                    "file_path": file["path"],
                    "language": language,
                    "type": "file",
                    "name": file_name,
                    "start_line": 1,
                    "end_line": total_lines,
                }
            }]

        chunks = []
        start = 0
        chunk_idx = 0

        while start < total_lines:
            end = min(start + self.max_lines, total_lines)
            chunk_content = "\n".join(lines[start:end])
            chunks.append({
                "chunk_id": f"{file['path']}::block::{chunk_idx}",
                "content": chunk_content,
                "metadata": {
                    "file_path": file["path"],
                    "language": language,
                    "type": "block",
                    "name": f"{file_name}:{start + 1}-{end}",
                    "start_line": start + 1,
                    "end_line": end,
                }
            })
            if end >= total_lines:
                break
            start += max(1, self.max_lines - self.line_overlap)
            chunk_idx += 1

        return chunks