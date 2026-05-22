import os
from pathlib import Path

from dotenv import load_dotenv

from rag_mcp.config import ROOT


def load_project_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    if os.environ.get("EMBEDDING_URL"):
        os.environ["embedding_url_override"] = os.environ["EMBEDDING_URL"]
