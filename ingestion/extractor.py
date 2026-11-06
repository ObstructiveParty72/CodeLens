class CodeExtractor:

    def extract(self, tree, source_code):
        results = []

        self._walk(tree.root_node,results,source_code)
        return results

    def _walk(self, node, results, source_code):
        # Convert source code to bytes
        source_bytes = source_code.encode("utf-8")

        # --------------------------------------------------
        # Function
        # --------------------------------------------------
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")

                results.append({
                    "type": "function",
                    "name": name_node.text.decode("utf-8"),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "content": content,
                })

        # --------------------------------------------------
        # Class
        # --------------------------------------------------
        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")

                results.append({
                    "type": "class",
                    "name": name_node.text.decode("utf-8"),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "content": content,
                })

        # --------------------------------------------------
        # Import
        # --------------------------------------------------
        elif node.type in ("import_statement","import_from_statement"):
            content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")

            results.append({
                "type": "import",
                "name": content,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "content": content,
            })

        # --------------------------------------------------
        # Visit child nodes
        # --------------------------------------------------
        for child in node.children:
            self._walk(child, results,source_code)