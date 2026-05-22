import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from rag_mcp.pipeline import RagPipeline


CORPUS_ENV = "RAG_CORPUS_PATH"


mcp = FastMCP("rag-local-mcp")


def _pipeline() -> RagPipeline:
    root = os.environ.get(CORPUS_ENV, "")
    if root == "":
        raise RuntimeError("RAG_CORPUS_PATH is not set")
    return RagPipeline(Path(root))


@mcp.tool()
def rag_sync_index() -> dict[str, int]:
    """Synchronise local vector and BM25 indexes with corpus folder."""
    return _pipeline().sync_index()


@mcp.tool()
def rag_search(query: str, use_hyde: bool = True) -> list[dict]:
    """Hybrid retrieval over local corpus."""
    pipeline = _pipeline()
    search_text = pipeline.hyde.expand(query) if use_hyde else query
    hits = pipeline.searcher.search(search_text)
    return [hit.model_dump(mode="json") for hit in hits]


@mcp.tool()
def rag_answer(query: str, use_hyde: bool = True) -> dict:
    """Answer user query with citations from local RAG store."""
    result = _pipeline().query(query, use_hyde=use_hyde)
    return result.model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
