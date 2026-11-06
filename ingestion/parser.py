from tree_sitter import Language, Parser

class CodeParser:
    def __init__(self):
        self.languages = {}

        try:
            import tree_sitter_python
            self.languages["python"] = Language(tree_sitter_python.language())
        except Exception:
            pass

        try:
            import tree_sitter_javascript
            self.languages["javascript"] = Language(tree_sitter_javascript.language())
        except Exception:
            pass

        try:
            import tree_sitter_typescript
            self.languages["typescript"] = Language(tree_sitter_typescript.language_typescript())
        except Exception:
            pass

    def is_supported(self, language_name: str) -> bool:
        return language_name in self.languages

    def parse(self, code: str, language_name: str):
        if not self.is_supported(language_name):
            raise ValueError(f"Unsupported parser language: {language_name}")
        language = self.languages[language_name]
        parser = Parser(language)
        tree = parser.parse(code.encode("utf-8"))
        return tree