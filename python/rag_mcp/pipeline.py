from pathlib import Path

from rag_mcp.hybrid_search import HybridSearcher
from rag_mcp.hyde import HydeExpander
from rag_mcp.indexer import RagIndexer
from rag_mcp.llm_zveno import ZvenoClient
from rag_mcp.schemas import RagAnswer, SearchResult


ANSWER_SYSTEM = (
    "Ты отвечаешь строго по предоставленным фрагментам документов на русском языке. "
    "Если данных недостаточно, напиши: «В предоставленных фрагментах ответ не найден.» "
    "Не выдумывай факты."
)


class RagPipeline:
    def __init__(self, corpus_root: Path):
        self.corpus_root = corpus_root
        self.indexer = RagIndexer(corpus_root)
        self.searcher = HybridSearcher(corpus_root)
        self.hyde = HydeExpander()
        self.llm = ZvenoClient()

    def sync_index(self) -> dict[str, int]:
        return self.indexer.sync()

    def query(self, user_query: str, use_hyde: bool = True) -> RagAnswer:
        search_text = self.hyde.expand(user_query) if use_hyde else user_query
        hits = self.searcher.search(search_text)
        if len(hits) == 0:
            return RagAnswer(
                answer="В предоставленных фрагментах ответ не найден.",
                sources=[],
                found=False,
            )
        context = self._build_context(hits)
        answer = self.llm.chat(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {user_query}\n\n"
                        f"Фрагменты:\n{context}\n\n"
                        "Сформируй ответ."
                    ),
                },
            ],
            max_tokens=900,
        )
        found = "не найден" not in answer.lower()
        return RagAnswer(answer=answer, sources=hits, found=found)

    def _build_context(self, hits: list[SearchResult]) -> str:
        blocks = []
        for idx, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{idx}] файл={hit.file_path}\n{hit.text}"
            )
        return "\n\n".join(blocks)
