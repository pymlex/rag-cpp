import faulthandler
import os

import gradio as gr
from dotenv import load_dotenv

from python.corpus_paths import get_corpus_root
from python.index_manager import IndexManager
from python.rag_pipeline import RagPipeline


faulthandler.enable()
load_dotenv()

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]


def _sync_status(stats: dict) -> str:
    return (
        f"добавлено {stats['added']}, обновлено {stats['updated']}, "
        f"удалено {stats['removed']}, пересборка {stats['rebuilt']}"
    )


def run_sync(progress: gr.Progress = gr.Progress()) -> str:
    progress(0.05, desc="Сканирование корпуса")
    manager = IndexManager(get_corpus_root())
    progress(0.2, desc="Чанки и эмбеддинги")
    stats = manager.sync()
    manager.close()
    progress(1.0, desc="Индекс обновлён")
    return _sync_status(stats)


def chat_respond(
    user_message: str,
    history: list,
    use_hyde: bool,
    progress: gr.Progress = gr.Progress(),
):
    if not user_message.strip():
        yield history, ""
        return
    history = history + [[user_message, None]]
    manager = IndexManager(get_corpus_root())
    pipeline = RagPipeline(manager)
    answer = ""
    for event in pipeline.answer_events(user_message, use_hyde=use_hyde, sync_first=False):
        if len(event) == 2:
            label, frac = event
            progress(frac, desc=label)
            yield history, label
            continue
        _, _, _, answer = event
    manager.close()
    history[-1][1] = answer
    progress(1.0, desc="Готово")
    yield history, ""


def clear_chat() -> tuple[list, str]:
    return [], ""


corpus_label = os.environ.get("RAG_CORPUS_ROOT", "")

with gr.Blocks(title="RAG", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        corpus_path = gr.Textbox(
            value=corpus_label,
            label="Корпус",
            interactive=False,
            scale=4,
        )
        use_hyde = gr.Checkbox(label="HyDE", value=True, scale=0, min_width=80)
    status_line = gr.Markdown("")
    chatbot = gr.Chatbot(
        height=560,
        show_copy_button=True,
        latex_delimiters=LATEX_DELIMITERS,
        render_markdown=True,
        bubble_full_width=False,
    )
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Сообщение",
            show_label=False,
            scale=8,
            lines=2,
        )
        send_btn = gr.Button("Отправить", variant="primary", scale=1, min_width=120)
    with gr.Row():
        sync_btn = gr.Button("Обновить индекс", scale=1)
        clear_btn = gr.Button("Очистить чат", scale=1)
        sync_result = gr.Textbox(show_label=False, interactive=False, scale=3)

    send_btn.click(
        chat_respond,
        [user_input, chatbot, use_hyde],
        [chatbot, status_line],
    ).then(lambda: "", None, user_input)
    user_input.submit(
        chat_respond,
        [user_input, chatbot, use_hyde],
        [chatbot, status_line],
    ).then(lambda: "", None, user_input)
    clear_btn.click(clear_chat, None, [chatbot, status_line])
    sync_btn.click(run_sync, None, sync_result, show_progress="full")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        max_threads=40,
    )
