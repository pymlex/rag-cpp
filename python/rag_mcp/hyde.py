from rag_mcp.config import CONFIG
from rag_mcp.llm_zveno import ZvenoClient


HYDE_SYSTEM = (
    "Ты переформулируешь короткий неформальный вопрос в один абзац "
    "научно-технического стиля для поиска по корпусу документов. "
    "Сохраняй смысл, не выдумывай факты, не добавляй ответ."
)


class HydeExpander:
    def __init__(self, llm: ZvenoClient | None = None):
        self.llm = llm or ZvenoClient()

    def expand(self, query: str) -> str:
        prompt = (
            f"Вопрос пользователя: {query}\n"
            "Верни только переформулированный поисковый запрос."
        )
        text = self.llm.chat(
            [
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=220,
        )
        if len(text) > CONFIG["hyde_max_chars"]:
            return text[: CONFIG["hyde_max_chars"]]
        return text
