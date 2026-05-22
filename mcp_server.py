from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from python.mcp_server import mcp


if __name__ == "__main__":
    mcp.run()
