import os

import gradio as gr
from dotenv import load_dotenv

from python.corpus_paths import get_corpus_root
from python.index_manager import IndexManager
from python.rag_pipeline import RagPipeline


load_dotenv()


def run_query(query: str, use_hyde: bool) -> tuple[str, str, str]:
    root = get_corpus_root()
    manager = IndexManager(root)
    pipeline = RagPipeline(manager)
    stats = manager.sync()
    hyde_text, answer = pipeline.answer(query, use_hyde=use_hyde, sync_first=False)
    manager.close()
    sync_info = (
        f"corpus={root} added={stats['added']} updated={stats['updated']} "
        f"removed={stats['removed']} rebuilt={stats['rebuilt']}"
    )
    return sync_info, hyde_text, answer


with gr.Blocks(title="RAG MCP Local") as demo:
    gr.Markdown(f"Корпус: `{os.environ.get('RAG_CORPUS_ROOT', '')}` (задаётся в `.env`)")
    query = gr.Textbox(label="Запрос", lines=3)
    hyde = gr.Checkbox(label="HyDE", value=True)
    sync_out = gr.Textbox(label="Синхронизация индекса", interactive=False)
    hyde_out = gr.Textbox(label="Текст для векторизации", interactive=False)
    answer = gr.Textbox(label="Ответ", lines=12, interactive=False)
    btn = gr.Button("Запрос")
    btn.click(run_query, [query, hyde], [sync_out, hyde_out, answer])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
