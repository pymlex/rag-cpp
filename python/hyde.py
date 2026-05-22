from python.llm_client import ZvenoClient
from config.settings import HYDE_MAX_TOKENS, LLM_TEMPERATURE


HYDE_SYSTEM = (
    "Ты переформулируешь короткий вопрос пользователя в одно научно-техническое "
    "предложение на русском языке для семантического поиска по корпусу документов. "
    "Не добавляй ответ на вопрос. Не выдумывай факты. Верни только одно предложение."
)


def expand_query(client: ZvenoClient, query: str) -> str:
    text = client.chat(
        messages=[
            {"role": "system", "content": HYDE_SYSTEM},
            {"role": "user", "content": query},
        ],
        max_tokens=HYDE_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    merged = f"{query.strip()} {text.strip()}".strip()
    if len(merged) > 4000:
        return merged[:4000]
    return merged
