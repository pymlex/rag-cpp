import faulthandler
import os
import sys

import gradio as gr
from dotenv import load_dotenv

from python.corpus_paths import get_corpus_root
from python.index_manager import IndexManager
from python.rag_pipeline import RagPipeline


faulthandler.enable()
load_dotenv()


def _format_sync(stats: dict) -> str:
    root = get_corpus_root()
    return (
        f"corpus={root} added={stats['added']} updated={stats['updated']} "
        f"removed={stats['removed']} rebuilt={stats['rebuilt']}"
    )


def run_sync() -> str:
    print("sync start", flush=True)
    manager = IndexManager(get_corpus_root())
    stats = manager.sync()
    manager.close()
    print("sync done", flush=True)
    return _format_sync(stats)


def run_query(query: str, use_hyde: bool) -> tuple[str, str]:
    print("query start", flush=True)
    manager = IndexManager(get_corpus_root())
    pipeline = RagPipeline(manager)
    hyde_text, answer = pipeline.answer(query, use_hyde=use_hyde, sync_first=False)
    manager.close()
    print("query done", flush=True)
    return hyde_text, answer


with gr.Blocks(title="RAG MCP Local") as demo:
    gr.Markdown(f"Корпус: `{os.environ.get('RAG_CORPUS_ROOT', '')}` (задаётся в `.env`)")
    gr.Markdown(
        "Сначала «Синхронизировать индекс» (может занять много времени). "
        "Затем введите вопрос в «Запрос» и нажмите «Ответ»."
    )
    sync_out = gr.Textbox(label="Синхронизация индекса", interactive=False)
    btn_sync = gr.Button("Синхронизировать индекс")
    query = gr.Textbox(label="Запрос", lines=3)
    hyde = gr.Checkbox(label="HyDE", value=True)
    hyde_out = gr.Textbox(label="Текст для векторизации", interactive=False)
    answer = gr.Textbox(label="Ответ", lines=12, interactive=False)
    btn_query = gr.Button("Ответ")
    btn_sync.click(run_sync, None, sync_out)
    btn_query.click(run_query, [query, hyde], [hyde_out, answer])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        max_threads=40,
    )
