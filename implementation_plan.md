# Replace In-Memory ChromaDB with pgvector

## What This Change Accomplishes

Right now your `hr_policy.pdf` is chunked, embedded, and stored in an **in-memory ChromaDB** instance every single time the server boots. The moment the server restarts — during development, deployment, or crash recovery — the entire vector database vanishes and is rebuilt from scratch.

This migration replaces ChromaDB with **pgvector**, a PostgreSQL extension that lets you store vector embeddings as a column in a regular SQL table. Since your backend already uses PostgreSQL for users, conversations, and messages, you will end up with a **single unified database** instead of two systems running side by side.

---

## What Changes in Plain English

| What | Before | After |
| :--- | :--- | :--- |
| **Vector Storage** | In-memory ChromaDB (lost on restart) | `policy_chunks` table in PostgreSQL (permanent) |
| **Ingestion** | Runs every single server boot | Runs **once** via a one-time `ingest.py` script |
| **Search** | ChromaDB `.query()` | PostgreSQL `ORDER BY embedding <=> :vector LIMIT 3` |
| **Server Startup** | Slow (PDF read + embed compute) | Instant (just connects to DB) |
| **Number of DBs** | 2 (PostgreSQL + ChromaDB) | 1 (PostgreSQL only) |
| **Dependencies** | `chromadb` | `pgvector` |

---

## User Review Required

> [!IMPORTANT]
> This migration requires enabling the `pgvector` extension on your PostgreSQL database. Run this **once** as a superuser before starting the server:
> ```sql
> CREATE EXTENSION IF NOT EXISTS vector;
> ```
> If you are using a hosted PostgreSQL service (e.g. Supabase, Neon, Railway), pgvector is already enabled by default.

> [!WARNING]
> After this migration, the `chromadb` package is no longer needed. The `policy_chunks` table will be created automatically on first server start. You must also run `ingest.py` **once** after the table is created to populate the PDF embeddings before the chatbot will work.

---

## Proposed Changes

### 1. Dependencies

---

#### [MODIFY] [requirements.txt](file:///c:/Users/Parthiv/Desktop/OEAI/HRBOT/requirements.txt)
- **Remove:** `chromadb`
- **Add:** `pgvector>=0.3.0`

The `pgvector` package provides the SQLAlchemy `Vector(n)` column type and cosine distance helpers used in the ORM query.

---

### 2. Database Layer

---

#### [MODIFY] [database.py](file:///c:/Users/Parthiv/Desktop/OEAI/HRBOT/database.py)

Add one new SQLAlchemy ORM model called **`PolicyChunk`**. This is the new permanent home for your PDF chunks and their embeddings.

```python
# New table that replaces the in-memory ChromaDB collection
class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id:  Mapped[str] = mapped_column(String(32), unique=True, index=True)  # e.g. "doc_001"
    topic:     Mapped[str] = mapped_column(String(255))                           # e.g. "HR Policy Chunk 1"
    content:   Mapped[str] = mapped_column(Text)                                  # The raw chunk text
    embedding: Mapped[list] = mapped_column(Vector(384))                          # 384 floats from all-MiniLM-L6-v2
```

Also add a helper function `search_similar_chunks(query_embedding, top_k)` to `database.py` that runs the pgvector cosine-distance query and returns matching chunks. This keeps all SQL logic in `database.py` — clean separation of concerns.

Also update `init_db()` to:
1. Enable the `vector` extension via `CREATE EXTENSION IF NOT EXISTS vector`.
2. Create the `policy_chunks` table.
3. Create an **HNSW cosine index** on the `embedding` column so searches run fast even with thousands of chunks.

---

### 3. Ingestion Script

---

#### [NEW] `ingest.py` — One-Time PDF Ingestion Script

This is a brand-new standalone script you run **once** to populate the `policy_chunks` table from `hr_policy.pdf`. You never need to run it again unless you change the PDF.

```
python ingest.py
```

**What it does:**
1. Reads `hr_policy.pdf` using PyMuPDF (same as before).
2. Splits the text into 200-word chunks (same chunking logic as before).
3. Encodes all chunks using `SentenceTransformer("all-MiniLM-L6-v2")`.
4. Inserts each chunk into the `policy_chunks` table with `INSERT ... ON CONFLICT DO NOTHING` (so re-running it is safe and won't duplicate data).

After this runs once, the embeddings live permanently in PostgreSQL.

---

### 4. Agent Logic

---

#### [MODIFY] [agent.py](file:///c:/Users/Parthiv/Desktop/OEAI/HRBOT/agent.py)

This is the biggest change. Three things are modified:

**a) Remove ChromaDB builder functions:**
- `build_chromadb()` — deleted entirely (no longer needed).
- `test_retrieval()` — deleted (ChromaDB-specific test harness).

**b) Update `HRAgent.__init__`:**
- Currently: `def __init__(self, llm, embedder, collection)` — takes a ChromaDB `collection` object.
- After: `def __init__(self, llm, embedder)` — no collection argument; retrieval is done via SQL through `database.py`.

**c) Rewrite `retrieval_node`:**

Currently the retrieval node encodes the user question and calls `self.collection.query(...)`. After the change it will:
1. Encode the user's question into a 384-dimension vector using `self.embedder`.
2. Call `search_similar_chunks(query_embedding, top_k=3)` from `database.py`.
3. Format the results into the same `context` and `sources` variables as before so the rest of the graph is completely unchanged.

The router, memory, answer, eval, and save nodes are **completely untouched**.

---

### 5. Server Startup

---

#### [MODIFY] [server.py](file:///c:/Users/Parthiv/Desktop/OEAI/HRBOT/server.py)

The `lifespan()` startup function currently does this on every boot:

```python
# BEFORE — slow, runs every single restart
documents  = load_documents_from_pdf(PDF_PATH)  # reads the PDF
collection = build_chromadb(documents, embedder)  # builds Chroma in RAM
agent_state["agent"] = HRAgent(llm, embedder, collection)
```

After the change:

```python
# AFTER — instant, chunks already stored in PostgreSQL
agent_state["agent"] = HRAgent(llm, embedder)   # no collection arg needed
```

Also remove the imports of `load_documents_from_pdf` and `build_chromadb` from `agent.py` since they no longer exist.

---

### 6. Streamlit Interface

---

#### [MODIFY] [capstone_streamlit.py](file:///c:/Users/Parthiv/Desktop/OEAI/HRBOT/capstone_streamlit.py)

Same change as `server.py` — remove the ChromaDB setup calls from the `@st.cache_resource` initialisation function so startup is instant there too.

---

## Verification Plan

### Automated Tests
- The existing `run_tests()` in `agent.py` covers 12 test cases including leave policy, WFH, red-team prompt injection, tool usage, and out-of-scope queries. All 12 must still pass after the migration.

### Manual Verification
1. Run `python ingest.py` — verify it prints `"Inserted X chunks into policy_chunks."`.
2. Run `uvicorn server:app --reload` — verify server starts **without** any PDF loading or ChromaDB messages.
3. Hit `/health` — verify `agent_ready: true`.
4. Send a chat message about leave policy via the React UI — verify the answer is grounded and `sources` are returned.
5. Restart the server and re-send the same question — verify the answer is identical (confirming embeddings are persisted).
