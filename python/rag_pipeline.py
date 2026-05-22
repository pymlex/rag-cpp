from collections.abc import Callable

import numpy as np

from config.settings import FINAL_TOP_K, RETRIEVAL_TOP_K
from python.embeddings_client import EmbeddingsClient
from python.hyde import expand_query
from python.hybrid_search import HybridSearcher
from python.index_manager import IndexManager
from python.llm_client import ZvenoClient
from config.settings import LLM_MAX_OUTPUT_TOKENS, LLM_TEMPERATURE


RAG_SYSTEM = (
    "Ты отвечаешь на русском языке по предоставленным фрагментам корпуса. "
    "Если в фрагментах нет данных для ответа, напиши: "
    "«В предоставленных документах нет информации для ответа на этот вопрос.» "
    "Не выдумывай факты."
)


class RagPipeline:
    def __init__(self, manager: IndexManager):
        self.manager = manager
        self.llm = ZvenoClient()
        self.embedder = EmbeddingsClient()

    def _searcher(self) -> HybridSearcher:
        return HybridSearcher(
            self.manager.vector_index,
            self.manager.bm25,
            self.manager.meta,
            self.manager.vector_id_to_chunk,
        )

    def answer(
        self,
        query: str,
        use_hyde: bool = True,
        sync_first: bool = False,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> tuple[str, str]:
        def step(label: str, frac: float) -> None:
            if on_progress is not None:
                on_progress(label, frac)

        if sync_first:
            self.manager.sync()
        if use_hyde:
            step("Расширение запроса", 0.12)
            search_text = expand_query(self.llm, query)
        else:
            step("Подготовка запроса", 0.12)
            search_text = query
        step("Векторизация", 0.28)
        vector = self.embedder.embed([search_text])[0]
        step("Гибридный поиск", 0.48)
        hits = self._searcher().search(vector, search_text, RETRIEVAL_TOP_K)
        picked = hits[:FINAL_TOP_K]
        step(f"Отобрано фрагментов: {len(picked)} из {len(hits)}", 0.62)
        context_blocks = []
        for idx, hit in enumerate(picked, start=1):
            context_blocks.append(
                f"[{idx}] {hit.file_path} (chunk {hit.chunk_index})\n{hit.text}"
            )
        context = "\n\n".join(context_blocks)
        if not context.strip():
            step("Контекст не найден", 1.0)
            return search_text, (
                "В предоставленных документах нет информации для ответа на этот вопрос."
            )
        step("Генерация ответа", 0.82)
        answer = self.llm.chat(
            messages=[
                {"role": "system", "content": RAG_SYSTEM},
                {
                    "role": "user",
                    "content": f"Вопрос: {query}\n\nКонтекст:\n{context}",
                },
            ],
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        step("Готово", 1.0)
        return search_text, answer

    def answer_events(
        self,
        query: str,
        use_hyde: bool = True,
        sync_first: bool = False,
    ):
        if sync_first:
            yield "Синхронизация индекса", 0.05
            self.manager.sync()
        if use_hyde:
            yield "Расширение запроса", 0.12
            search_text = expand_query(self.llm, query)
        else:
            yield "Подготовка запроса", 0.12
            search_text = query
        yield "Векторизация", 0.28
        vector = self.embedder.embed([search_text])[0]
        yield "Гибридный поиск", 0.48
        hits = self._searcher().search(vector, search_text, RETRIEVAL_TOP_K)
        picked = hits[:FINAL_TOP_K]
        yield f"Отобрано фрагментов: {len(picked)} из {len(hits)}", 0.62
        context_blocks = []
        for idx, hit in enumerate(picked, start=1):
            context_blocks.append(
                f"[{idx}] {hit.file_path} (chunk {hit.chunk_index})\n{hit.text}"
            )
        context = "\n\n".join(context_blocks)
        if not context.strip():
            yield "Контекст не найден", 1.0
            not_found = (
                "В предоставленных документах нет информации для ответа на этот вопрос."
            )
            yield "Готово", 1.0, search_text, not_found
            return
        yield "Генерация ответа", 0.82
        answer = self.llm.chat(
            messages=[
                {"role": "system", "content": RAG_SYSTEM},
                {
                    "role": "user",
                    "content": f"Вопрос: {query}\n\nКонтекст:\n{context}",
                },
            ],
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        yield "Готово", 1.0, search_text, answer
