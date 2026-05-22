from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from config.settings import FINAL_TOP_K
from python.corpus_paths import get_corpus_root
from python.index_manager import IndexManager
from python.rag_pipeline import RagPipeline


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

mcp = FastMCP("rag-local-mcp")


@mcp.tool()
def rag_sync_index() -> dict:
    manager = IndexManager(get_corpus_root())
    stats = manager.sync()
    manager.close()
    return stats


@mcp.tool()
def rag_query(query: str, use_hyde: bool = True) -> dict:
    manager = IndexManager(get_corpus_root())
    pipeline = RagPipeline(manager)
    manager.sync()
    hyde_text, answer = pipeline.answer(query, use_hyde=use_hyde)
    manager.close()
    return {"hyde_query": hyde_text, "answer": answer}


@mcp.tool()
def rag_search_chunks(query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
    from python.embeddings_client import EmbeddingsClient
    from python.hybrid_search import HybridSearcher
    from python.hyde import expand_query
    from python.llm_client import ZvenoClient

    manager = IndexManager(get_corpus_root())
    manager.sync()
    llm = ZvenoClient()
    embedder = EmbeddingsClient()
    search_text = expand_query(llm, query)
    vector = embedder.embed([search_text])[0]
    searcher = HybridSearcher(
        manager.vector_index,
        manager.bm25,
        manager.meta,
        manager.vector_id_to_chunk,
    )
    hits = searcher.search(vector, search_text, top_k)
    manager.close()
    return [
        {
            "chunk_id": h.chunk_id,
            "file_path": h.file_path,
            "chunk_index": h.chunk_index,
            "text": h.text[:1200],
        }
        for h in hits
    ]


if __name__ == "__main__":
    mcp.run()
