import os

from openai import OpenAI

from rag_mcp.config import CONFIG
from rag_mcp.env_loader import load_project_env


load_project_env()


class ZvenoClient:
    def __init__(self):
        api_key = os.environ["ZVENOAI_API_KEY"]
        self.client = OpenAI(
            api_key=api_key,
            base_url=CONFIG["zveno_base_url"],
        )
        self.model = CONFIG["zveno_model"]

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 900) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
