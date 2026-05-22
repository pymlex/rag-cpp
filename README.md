# Local RAG with C++ HNSW and MCP

## Overview

This project is a Windows-first local retrieval stack for scientific and engineering text. A custom C++ HNSW index stores `float32` embeddings with cosine distance. Python handles chunking, hybrid retrieval, HyDE query expansion, Zveno LLM answering, Gradio UI, and an MCP server compatible with Cursor and other MCP clients.

The embedding model is served locally through [embeddings-inference-server](https://github.com/pymlex/embeddings-inference-server) with `mlsa-iai-msu-lab/sci-rus-tiny` (312-dimensional vectors, max sequence length 1024 tokens).

Repository: [pymlex/rag-cpp](https://github.com/pymlex/rag-cpp)

## Stack

| Layer | Role |
| --- | --- |
| C++ `ragdb_native` | HNSW graph, persistence, pybind11 module |
| SQLite | Chunk text, metadata, `vector_blob` for hard rebuild |
| BM25 | Lexical retrieval (`rank-bm25`) |
| Hybrid merge | `0.65` vector + `0.35` BM25 after min–max normalisation |
| HyDE | Original user query plus one LLM paraphrase before embedding |
| Zveno API | `openai/gpt-oss-120b` via OpenAI-compatible HTTP |

Chunking uses `500` tokens per chunk and `100` token overlap (same tokenizer family as the embedding model).

## Prerequisites

- Windows 10
- **Python 3.10 or newer for `.venv`** (the `mcp` package does not support 3.9)
- `git`
- `cmake` 3.20+
- MinGW `gcc` / `g++` (detected by the build script) or MSVC toolchain
- Internet access for the first install (pip, Hugging Face model weights, Zveno API, embedding server clone)

Run the system check before any build:

```powershell
git clone https://github.com/pymlex/rag-cpp.git
cd rag-cpp
.\scripts\check_system.ps1
```

Save the printed output if the C++ build fails later. The line `Python 3.10+ for venv` must point to an interpreter, not `NOT FOUND`.

If `python --version` on PATH is 3.9 but Python 3.12 is installed, the install script still works: `scripts\resolve_python.ps1` picks `py -3.12` or `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`.

Manual override:

```powershell
.\scripts\install_all.ps1 -PythonExe "C:\Users\aleks\AppData\Local\Programs\Python\Python312\python.exe"
```

If a previous install created `.venv` with Python 3.9, delete it before reinstall:

```powershell
Remove-Item -Recurse -Force .venv
```

## End-to-end installation

### 1. Create the virtual environment and install Python packages

```powershell
cd rag-cpp
.\scripts\install_all.ps1
```

The script performs, in order:

1. `scripts\check_system.ps1`
2. `.venv` creation
3. `pip install -r requirements.txt`
4. copy `.env.example` to `.env` when `.env` is missing
5. C++ build through `scripts\build_cpp.ps1` (MinGW generator when `gcc` is found)
6. clone `third_party\embeddings-inference-server` and install its `requirements.txt`

If the C++ step fails, install MinGW or MSVC, rerun `.\scripts\build_cpp.ps1`, then continue from step 2 below.

To skip the native build during install:

```powershell
.\scripts\install_all.ps1 -SkipCpp
.\scripts\build_cpp.ps1
```

### 2. Configure environment variables

Open `.env` in the repository root. It is created from `.env.example` on first install. Never commit `.env`.

| Variable | Purpose |
| --- | --- |
| `ZVENOAI_API_KEY` | Bearer token for [Zveno](https://api.zveno.ai/v1) |
| `ZVENO_BASE_URL` | API base, default `https://api.zveno.ai/v1` |
| `ZVENO_MODEL` | LLM id, default `openai/gpt-oss-120b` |
| `EMBEDDING_URL` | Local embedding HTTP root, default `http://127.0.0.1:8000/` |
| `RAG_CORPUS_ROOT` | Single folder with all documents (nested folders allowed) |
| `EMBEDDING_MODEL_ID` | HF model id for chunk tokenisation, default `mlsa-iai-msu-lab/sci-rus-tiny` |
| `RAG_PROFILE_CORPUS` | Optional override for py-spy scripts; usually same as `RAG_CORPUS_ROOT` |
| `PYSPY_DURATION` | Gradio profiling duration in seconds, default `30` |

Example `.env`:

```env
ZVENOAI_API_KEY=sk-your-real-key
ZVENO_BASE_URL=https://api.zveno.ai/v1
ZVENO_MODEL=openai/gpt-oss-120b
EMBEDDING_URL=http://127.0.0.1:8000/
RAG_CORPUS_ROOT=C:\data\my_corpus
EMBEDDING_MODEL_ID=mlsa-iai-msu-lab/sci-rus-tiny
RAG_PROFILE_CORPUS=C:\data\my_corpus
PYSPY_DURATION=30
```

Put only text files you want indexed inside `RAG_CORPUS_ROOT`. Supported extensions are listed in `config/settings.py` (`.md`, `.py`, `.cpp`, `.txt`, `.json`, and others). Images, audio, and binaries are skipped.

The index is written to `RAG_CORPUS_ROOT\rag_index\` (`hnsw.bin`, `chunks.sqlite`, `bm25_corpus.json`). That directory is gitignored.

### 3. Activate the virtual environment

Every new terminal session:

```powershell
cd rag-cpp
.\.venv\Scripts\Activate.ps1
```

### 4. Start the embedding server

Terminal A:

```powershell
.\scripts\run_embeddings_local.ps1
```

This sets `MODEL_DIR=mlsa-iai-msu-lab/sci-rus-tiny` and runs `uvicorn` on `http://127.0.0.1:8000` without ngrok. The first start downloads model weights from Hugging Face.

Quick check:

```powershell
curl -Method POST -Uri "http://127.0.0.1:8000/" -ContentType "application/json" -Body '{"inputs":["тест"]}'
```

You should receive a JSON array of one 312-dimensional vector.

### 5. Index the corpus and open Gradio

Terminal B (venv active, embedding server still running):

```powershell
python gradio_app.py
```

Open `http://127.0.0.1:7860`. The UI shows the corpus path from `RAG_CORPUS_ROOT`. Each query triggers a sync (new, changed, and deleted files), HyDE expansion, hybrid retrieval, and an LLM answer. If nothing relevant is found, the model returns an explicit not-found message in Russian.

CLI one-shot query:

```powershell
python main.py "ваш вопрос"
```

## MCP integration

Copy `mcp_config.example.json` into your Cursor MCP settings and adjust paths.

```json
{
  "mcpServers": {
    "rag-local": {
      "command": "C:\\path\\to\\rag-cpp\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\rag-cpp\\mcp_server.py"],
      "env": {
        "RAG_CORPUS_ROOT": "C:\\path\\to\\corpus",
        "EMBEDDING_URL": "http://127.0.0.1:8000/",
        "MINGW_DLL_DIR": "C:\\msys64\\ucrt64\\bin"
      }
    }
  }
}
```

Use the launcher `mcp_server.py` in the repo root, not `python\\mcp_server.py`. `ZVENOAI_API_KEY` is read from `.env` in the repo root; do not duplicate secrets in the MCP JSON.

Tools:

| Tool | Action |
| --- | --- |
| `rag_sync_index` | Scan corpus, update SQLite, BM25, and HNSW |
| `rag_query` | Full RAG answer with optional HyDE |
| `rag_search_chunks` | Return top chunks without final generation |

Manual server start:

```powershell
python mcp_server.py
```

## Index update semantics

On each sync the scanner compares file hashes. Unchanged files are skipped. Removed files are deleted from SQLite; the HNSW file is rebuilt from remaining `vector_blob` rows (hard delete, no tombstones). Updated files are re-chunked and trigger the same rebuild path. Pure additions use incremental HNSW inserts only.

## Hybrid retrieval

For query $q$, let $s_v$ be the normalised cosine score and $s_b$ the normalised BM25 score for chunk $c$:

$$
s(c) = 0.65 \cdot s_v(c) + 0.35 \cdot s_b(c)
$$

Top chunks are passed to the LLM with file path and chunk index.

## Benchmark pipeline

Requirements: `.env` filled, embedding server running, `RAG_CORPUS_ROOT` populated with your documents.

### 1. Generate 20 QA pairs with Zveno

```powershell
.\.venv\Scripts\Activate.ps1
python benchmarks\generate_qa_dataset.py
```

Output: `benchmarks/generated_qa.json` (also archived under `benchmarks/datasets/`).

### 2. Run retrieval and answer evaluation

```powershell
python benchmarks\run_local_rag_eval.py
```

Or both steps:

```powershell
.\scripts\run_benchmark.ps1
```

Metrics written to `benchmarks/results/<timestamp>/metrics.json`:

| Metric | Definition |
| --- | --- |
| `retrieval_pass_rate` | Share of cases where `source_file` appears in hybrid top-8 |
| `answer_pass_rate` | Share of cases where the RAG answer contains at least half of `must_contain` phrases |
| `combined_pass_rate` | Fraction passing both checks |

Plots: `pass_rates.png`, `aggregate.png`.

### 3. Optional τ²-bench

Installed from GitHub during `install_all.ps1` (`requirements-benchmark.txt`). Manual install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
python benchmarks\run_tau2_rag.py
```

Uses the [τ²-bench](https://github.com/sierra-research/tau2-bench) CLI. Logs land in `benchmarks/results/`.

## Profiling with py-spy

```powershell
$env:RAG_PROFILE_CORPUS = $env:RAG_CORPUS_ROOT
python profiling\run_pyspy_index.py
python profiling\run_pyspy_gradio.py
```

Flame graphs are stored under `profiling/results/`.

## Project layout

```
rag-cpp/
├── cpp/                 # HNSW implementation and CMake build
├── python/              # Index manager, RAG, MCP, hybrid search
├── config/settings.py   # Extensions, chunk sizes, hybrid weights
├── benchmarks/          # QA generation and evaluation
├── profiling/           # py-spy runners
├── scripts/             # install, build, embedding server, benchmark
├── uml/                 # PlantUML architecture diagrams
├── demo_corpus/         # Small sample documents
├── gradio_app.py
├── main.py
└── .env.example
```

## UML

- `uml/architecture.puml` — component view
- `uml/query_flow.puml` — request sequence

Render with any PlantUML viewer for coursework figures.

## Rebuild native module only

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\build_cpp.ps1
```

The built `ragdb_native*.pyd` is copied into the repo root, `python/`, and `.venv\Lib\site-packages\`.

If `ModuleNotFoundError: ragdb_native` appears after a successful build:

```powershell
.\scripts\install_native_module.ps1
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No matching distribution found for mcp` | PATH Python is 3.9. Remove `.venv`, install Python 3.12, rerun `install_all.ps1`. |
| `pip uninstall` access denied | Fixed in install script: uses `python -m pip`, not `pip.exe`. |
| `No module named pybind11` in CMake | Run `install_all.ps1` fully; build uses `.venv` Python, not system 3.12 without packages. |
| `mingw32-make: No such file` | CMake configure failed first; read lines above `No rule to make target`. |
| `No matching distribution found for tau2-bench` | Package is not on PyPI; use `requirements-benchmark.txt` (git install). Core RAG works without it. |
| `install TARGETS given no LIBRARY DESTINATION` | Fixed in current `CMakeLists.txt`; run `git pull` and `.\scripts\build_cpp.ps1`. |
| `DLL load failed while importing ragdb_native` | MinGW runtime missing. Run `.\scripts\copy_mingw_runtime.ps1` and `.\scripts\install_native_module.ps1`, or set `MINGW_DLL_DIR` in `.env`. |
| CMake picks wrong Python | Pass `-PythonExe` to `install_all.ps1` or delete `.venv`. |

## Daily run checklist

1. `.\.venv\Scripts\Activate.ps1`
2. `.\scripts\run_embeddings_local.ps1` (terminal A)
3. `python gradio_app.py` or MCP client (terminal B)
4. Ensure `RAG_CORPUS_ROOT` in `.env` points to your document tree
