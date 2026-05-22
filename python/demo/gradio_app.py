import os
from pathlib import Path

import gradio as gr

from rag_mcp.pipeline import RagPipeline


def run_query(corpus_path: str, query: str, use_hyde: bool) -> tuple[str, str]:
    root = Path(corpus_path)
    pipeline = RagPipeline(root)
    stats = pipeline.sync_index()
    result = pipeline.query(query, use_hyde=use_hyde)
    sources = "\n\n".join(
        f"{idx}. {hit.file_path} (score={hit.fused_score:.4f})\n{hit.text[:500]}"
        for idx, hit in enumerate(result.sources, start=1)
    )
    meta = f"sync: added={stats['added']} updated={stats['updated']} removed={stats['removed']}"
    return f"{meta}\n\n{result.answer}", sources


with gr.Blocks(title="RAG MCP Local") as demo:
    gr.Markdown("# Локальный RAG (HNSW + BM25 + HyDE)")
    corpus = gr.Textbox(label="Папка корпуса", value=os.environ.get("RAG_CORPUS_PATH", ""))
    query = gr.Textbox(label="Запрос", lines=3)
    use_hyde = gr.Checkbox(label="HyDE", value=True)
    answer = gr.Textbox(label="Ответ", lines=12)
    sources = gr.Textbox(label="Источники", lines=14)
    btn = gr.Button("Спросить")
    btn.click(run_query, [corpus, query, use_hyde], [answer, sources])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
