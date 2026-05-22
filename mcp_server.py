from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from python.rag_mcp_app import mcp


if __name__ == "__main__":
    mcp.run()
