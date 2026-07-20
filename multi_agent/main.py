"""
main.py — Entry point for the Multi-Agent RAG Chatbot server.

Startup sequence:
  1. Load & index documents from docs/ into Chroma vector store
  2. Build the hybrid retriever (BM25 + vector + CrossEncoder)
  3. Create the FastAPI app
  4. Start Uvicorn

Run with:
  python -m multi_agent.main
"""

import os
from dotenv import load_dotenv

# Load environment variables before importing anything that uses them
_current_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

from multi_agent.config import SERVER_HOST, SERVER_PORT
from multi_agent.retrieval.ingestion import load_and_index_documents
from multi_agent.retrieval.retriever import build_retriever
from multi_agent.api import create_app

# ── Startup sequence ──────────────────────────────────────────────────────────
print("[STARTUP] Indexing documents...")
chunks    = load_and_index_documents()

print("[STARTUP] Building hybrid retriever...")
retriever = build_retriever(chunks)

print("[STARTUP] Creating FastAPI app...")
app = create_app(chunks, retriever)

print(f"[STARTUP] Ready — http://{SERVER_HOST}:{SERVER_PORT}")

if __name__ == "__main__":
    import uvicorn
    # Check for PORT environment variable set by hosting providers (Hugging Face Spaces, Render, etc.)
    port = int(os.environ.get("PORT", SERVER_PORT))
    uvicorn.run(
        "multi_agent.main:app",
        host=SERVER_HOST,
        port=port,
        reload=False,
    )
