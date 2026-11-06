from ingestion.chunker import CodeChunker

files = [
    {
        "path": "auth/service.py",
        "language": "python",

        "code": [
            {
                "type": "class",
                "name": "AuthService",
                "start_line": 5,
                "end_line": 10,
                "content": """class AuthService:
                                def authenticate(self, token):
                                    return token
                            """
            },
            {
                "type": "function",
                "name": "verify_token",
                "start_line": 12,
                "end_line": 14,
                "content": """def verify_token(token):
                                return token != ""
            """
            }
        ]
    }
]


chunker = CodeChunker()
chunks = chunker.create_chunks(files)

for chunk in chunks:
    print("=" * 60)
    print(f"CHUNK ID: {chunk['chunk_id']}")
    print("CONTENT:")
    print(chunk["content"])
    print("\nMETADATA:")
    print(chunk["metadata"])