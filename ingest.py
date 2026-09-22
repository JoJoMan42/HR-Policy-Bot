"""One-time script to ingest hr_policy.pdf into PostgreSQL via pgvector.

Run this ONCE after setting up the database:
    python ingest.py

Re-running is safe — existing chunks are skipped via ON CONFLICT DO NOTHING.
"""

import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────
PDF_PATH = "hr_policy.pdf"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200  # words per chunk (matches the original chunking strategy)
MIN_CHUNK_CHARS = 50


def load_and_chunk_pdf(pdf_path: str) -> list[dict]:
    """Read a PDF and split it into word-based chunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at '{pdf_path}'.")

    print(f"[INGEST] Reading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()

    words = full_text.split()
    documents = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) < MIN_CHUNK_CHARS:
            continue
        documents.append(
            {
                "chunk_id": f"doc_{i + 1:03}",
                "topic": f"HR Policy Chunk {i // CHUNK_SIZE + 1}",
                "text": chunk.strip(),
            }
        )

    print(f"[INGEST] Created {len(documents)} chunks from {pdf_path}.")
    return documents


def main():
    # Initialise the database (creates tables + pgvector extension).
    from database import init_db, SessionLocal, PolicyChunk

    init_db()

    # Load and chunk the PDF.
    documents = load_and_chunk_pdf(PDF_PATH)
    if not documents:
        raise ValueError("No chunks created — PDF parsing may have failed.")

    # Embed all chunks.
    print(f"[INGEST] Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.encode(texts).tolist()
    print(f"[INGEST] Generated {len(embeddings)} embeddings ({len(embeddings[0])} dimensions).")

    # Insert into PostgreSQL (skip duplicates).
    inserted = 0
    skipped = 0
    with SessionLocal.begin() as db:
        for doc, emb in zip(documents, embeddings):
            exists = db.query(PolicyChunk).filter_by(chunk_id=doc["chunk_id"]).first()
            if exists:
                skipped += 1
                continue
            db.add(
                PolicyChunk(
                    chunk_id=doc["chunk_id"],
                    topic=doc["topic"],
                    content=doc["text"],
                    embedding=emb,
                )
            )
            inserted += 1

    print(f"[INGEST] Done — inserted {inserted} chunks, skipped {skipped} duplicates.")
    print("[INGEST] You can now start the server with: uvicorn server:app --reload")


if __name__ == "__main__":
    main()
