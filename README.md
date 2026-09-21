# 🧠 HR Policy RAG Assistant

An enterprise-ready **Retrieval-Augmented Generation (RAG)** system and AI assistant that answers employee HR policy queries with strict document grounding, PostgreSQL session persistence, and self-correction guardrails.

---

## 📸 Preview

![App Screenshot](picture.png)

---

## ✨ Features

- 📄 **Strict Grounding**: Answers policy questions strictly from official company HR documents (`hr_policy.pdf`).
- 🔍 **Vector Search (ChromaDB)**: Chunks documents and performs fast semantic retrieval using `sentence-transformers` (`all-MiniLM-L6-v2`).
- 🤖 **LangGraph Agent Orchestration**: Stateful graph coordinating query routing, memory recall, vector retrieval, custom tools, and self-evaluation.
- 🛠️ **Built-in Tools**:
  - **Leave Calculator**: Calculates remaining privilege leave balances.
  - **Date & Time Utility**: Provides real-time date/time calculations.
- 📊 **Self-Evaluation & Guardrails**: Assesses answer faithfulness before delivery, triggering automatic retries on hallucination.
- 🗄️ **Full Database Persistence**: PostgreSQL integration with SQLAlchemy for user accounts, conversation sessions, message history, and RAG evaluation metrics.
- 🔐 **JWT Authentication**: User registration and login with `bcrypt` password hashing and secure token-based chat isolation.
- 💻 **Dual Interfaces**:
  - **React + Vite App**: Modern UI with real-time connection status and quick question chips.
  - **Streamlit Interface**: Lightweight dashboard for quick local testing and deployment.

---

## 🏗️ Architecture & Pipeline

```text
[User Query]
     │
     ▼
[Memory Node] ──► Reads context from PostgreSQL (Thread History)
     │
     ▼
[Router Node] ──┬──► "retrieve"    ──► [Retrieval Node] (ChromaDB Vector Search)
                ├──► "tool"        ──► [Tool Node] (Leave / Date Calculator)
                └──► "memory_only" ──► [Skip Retrieval] (Greetings / Context)
                                              │
                                              ▼
                                       [Answer Node] (Groq LLM Generation)
                                              │
                                              ▼
                                       [Eval Node] (Faithfulness Check)
                                        │         │
                   (Score < 0.7 & Retry < 2)       (Passed / Max Retries)
                                        ▼                 ▼
                                 [Answer Node]      [Save Node] ──► Persist to PostgreSQL
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 18, Vite, Vanilla CSS |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **AI / RAG** | LangGraph, LangChain, Groq API (`openai/gpt-oss-120b` / `llama-3.1-8b-instant`) |
| **Vector DB & Embeddings** | ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`), PyMuPDF (`fitz`) |
| **Database & Auth** | PostgreSQL, SQLAlchemy 2.0, PyJWT, Bcrypt |
| **Alternative UI** | Streamlit |

---

## 📂 Project Structure

```text
.
├── agent.py                  # Core LangGraph agent, ChromaDB RAG, and tools
├── database.py               # PostgreSQL models (User, Conversation, Message) & queries
├── server.py                 # FastAPI backend API with JWT authentication
├── capstone_streamlit.py     # Standalone Streamlit interface
├── hr_policy.pdf             # Source HR policy document
├── requirements.txt          # Python dependencies
├── picture.png               # UI screenshot
├── .env.example              # Sample environment variables
├── README.md                 # Project documentation
└── frontend/                 # React + Vite web application
    ├── src/
    │   ├── components/       # AuthScreen, ChatMessage, QuickQuestion
    │   ├── App.jsx           # Main chat application & session manager
    │   ├── App.css           # Styling & theme variables
    │   └── main.jsx          # React entry point
    ├── index.html
    └── package.json
```

---

## 🚀 Quick Start

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/JoJoMan42/HR-Policy-Bot.git
cd HR-Policy-Bot
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/hrbot
JWT_SECRET_KEY=your_super_secret_random_jwt_key
```

---

### 3. Create PostgreSQL Database

Create the database once in PostgreSQL:

```sql
CREATE DATABASE hrbot;
```

*(All required tables — `users`, `conversations`, and `messages` — are created automatically on API startup).*

---

### 4. Run the Application

#### A. Start the FastAPI Backend
```powershell
uvicorn server:app --reload --port 8000
```

#### B. Start the React Frontend (in a second terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

#### C. (Optional) Run Streamlit UI
```powershell
streamlit run capstone_streamlit.py
```

---

## 🔌 API Reference

| Endpoint | Method | Auth | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | Service readiness & agent status |
| `/api/auth/signup` | `POST` | None | Create new user account (`email`, `password`) |
| `/api/auth/login` | `POST` | None | Authenticate and receive JWT access token |
| `/api/chat` | `POST` | Bearer Token | Send message to HR agent & receive grounded answer |
| `/api/conversations/{thread_id}/history` | `GET` | Bearer Token | Retrieve user's previous conversation history |

---

## 👨‍💻 Author

**Parthiv Datta**  
*3rd Year B.Tech CSE*
