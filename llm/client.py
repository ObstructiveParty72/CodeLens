import inspect
import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        configured_provider = provider or os.getenv("LLM_PROVIDER")
        if configured_provider:
            self.provider = configured_provider.lower()
        elif gemini_key:
            self.provider = "gemini"
        elif groq_key:
            self.provider = "groq"
        else:
            self.provider = "gemini"

        if self.provider == "gemini":
            if not gemini_key:
                raise ValueError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in .env. "
                    "Get a key at https://aistudio.google.com"
                )
            from google import genai
            self.client = genai.Client(api_key=gemini_key)
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

        elif self.provider == "groq":
            if not groq_key:
                raise ValueError("GROQ_API_KEY is not set in .env")
            from groq import Groq
            self.client = Groq(api_key=groq_key)
            self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'gemini' or 'groq'.")

    def generate(self, question: str, context: str):
        system_prompt = inspect.cleandoc("""
            You are an AI software engineer specialized in understanding codebases.
            Answer the user's question using the provided repository context.

            Rules:
            1. Use the provided code context as the primary source of truth.
            2. Do not invent files, functions, classes, or behavior.
            3. If the context does not contain enough information, state that clearly.
            4. Mention relevant file paths and line numbers when available.
            5. Mention function, method, or class names when useful.
            6. Explain the implementation and control flow clearly.
            7. Keep the answer concise, technical, and accurate.
        """)

        user_prompt = f"Repository Context:\n{context}\n\nUser Question:\n{question}"

        if self.provider == "gemini":
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config
            )
            return response.text

        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0
            )
            return response.choices[0].message.content