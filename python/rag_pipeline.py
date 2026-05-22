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

    def answer(self, query: str, use_hyde: bool = True, sync_first: bool = False) -> tuple[str, str]:
        if sync_first:
            self.manager.sync()
        search_text = expand_query(self.llm, query) if use_hyde else query
        vector = self.embedder.embed([search_text])[0]
        hits = self._searcher().search(vector, search_text, RETRIEVAL_TOP_K)
        context_blocks = []
        for idx, hit in enumerate(hits[:FINAL_TOP_K], start=1):
            context_blocks.append(
                f"[{idx}] {hit.file_path} (chunk {hit.chunk_index})\n{hit.text}"
            )
        context = "\n\n".join(context_blocks)
        if not context.strip():
            return search_text, (
                "В предоставленных документах нет информации для ответа на этот вопрос."
            )
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
        return search_text, answer
