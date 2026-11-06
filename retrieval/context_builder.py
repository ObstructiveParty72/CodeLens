class ContextBuilder:
    def build(self, results):
        context_parts = []

        for result in results:
            payload = result.payload if hasattr(result, "payload") else getattr(result, "payload", {})
            if not payload:
                continue

            file_path = payload.get("file_path", "unknown")
            name = payload.get("name", "unknown")
            chunk_type = payload.get("type", "unknown")
            start_line = payload.get("start_line", "?")
            end_line = payload.get("end_line", "?")
            content = payload.get("content", "").strip()

            context = (
                f"--- Source: {file_path} (Lines {start_line}-{end_line}) ---\n"
                f"Element: {chunk_type} {name}\n"
                f"```\n{content}\n```"
            )

            context_parts.append(context)

        return "\n\n".join(context_parts)