import os
from typing import Sequence

from openai import OpenAI

from config.settings import ZVENO_BASE_URL, ZVENO_MODEL


class ZvenoClient:
    def __init__(self):
        api_key = os.environ["ZVENOAI_API_KEY"]
        base_url = os.getenv("ZVENO_BASE_URL", ZVENO_BASE_URL)
        self.model = os.getenv("ZVENO_MODEL", ZVENO_MODEL)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
