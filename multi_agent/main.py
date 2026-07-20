"""
main.py — Entry point for the Multi-Agent RAG Chatbot server.

Startup sequence:
  1. Uvicorn binds to port immediately (Render detects it right away)
  2. Document indexing & retriever build run via FastAPI lifespan
  3. App is ready to serve requests

Run with:
  uvicorn multi_agent.main:app --host 0.0.0.0 --port $PORT
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables before importing anything that uses them
_current_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

from multi_agent.config import SERVER_HOST, SERVER_PORT
from multi_agent.retrieval.ingestion import load_and_index_documents
from multi_agent.retrieval.retriever import build_retriever
from multi_agent.api import create_app_with_lifespan

# ── Lifespan: runs AFTER uvicorn binds to port ────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    print("[STARTUP] Indexing documents...")
    chunks = load_and_index_documents()

    print("[STARTUP] Building hybrid retriever...")
    retriever = build_retriever(chunks)

    # Store on app state so routes can access them
    app.state.chunks = chunks
    app.state.retriever = retriever

    port = int(os.environ.get("PORT", SERVER_PORT))
    print(f"[STARTUP] Ready — http://{SERVER_HOST}:{port}")
    yield
    print("[SHUTDOWN] Cleaning up...")

# ── Create app ────────────────────────────────────────────────────────────────
app = create_app_with_lifespan(lifespan)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", SERVER_PORT))
    uvicorn.run(
        "multi_agent.main:app",
        host=SERVER_HOST,
        port=port,
        reload=False,
    )
