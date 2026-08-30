# 🧠 HR Policy RAG Agent

An intelligent HR assistant built using **Retrieval-Augmented Generation (RAG)** that answers employee queries based strictly on company HR policies.

---

## 📸 Demo / Preview

![App Screenshot](https://github.com/JoJoMan42/HR-Policy-Bot/blob/master/picture.png)

---

## 🚀 Features

* 📄 **Accurate Policy Q&A**: Answers HR-related questions using official company policy documents.
* 🔍 **Vector Search (ChromaDB)**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) and ChromaDB for semantic retrieval.
* 🤖 **LangGraph + Groq**: Robust agent orchestration powered by `llama-3.1-8b-instant`.
* 🛠️ **Built-in Tools**:
  * Leave balance calculator
  * Working days & date calculations
* 📊 **Guardrails & Evaluation**: Faithfulness evaluation and response transparency metrics.

---

## 🧠 How It Works

1. **Document Ingestion**: HR policy PDF is parsed and split into structured chunks.
2. **Vector Indexing**: Text chunks are embedded and indexed into ChromaDB.
3. **Smart Routing**: LangGraph router determines whether to retrieve policies, call tools, or answer from memory.
4. **Context Generation**: LLM generates answers strictly grounded in retrieved policy context.
5. **Self-Correction**: Faithfulness scoring ensures answers don't hallucinate.

---

## 🏗️ Tech Stack

* **Python 3.10+**
* **LangChain & LangGraph**
* **Groq API** (`llama-3.1-8b-instant`)
* **Sentence Transformers** (`all-MiniLM-L6-v2`)
* **ChromaDB** (Vector Database)
* **PyMuPDF** (`fitz`)
* **Streamlit** (Web Interface)

---

## 📂 Project Structure

```
.
├── agent.py                  # Core LangGraph agent & RAG logic
├── capstone_streamlit.py     # Streamlit web interface
├── hr_policy.pdf             # Official HR policy document
├── requirements.txt          # Production dependencies
├── picture.png               # App screenshot
└── README.md                 # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/JoJoMan42/HR-Policy-Bot.git
cd HR-Policy-Bot
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## ▶️ Run Locally

```bash
streamlit run capstone_streamlit.py
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Fork or push this repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app, set repository to `JoJoMan42/HR-Policy-Bot`, and main file to `capstone_streamlit.py`.
4. In **Advanced Settings > Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```
5. Click **Deploy**!

---

## 👨‍💻 Author

**Parthiv Datta**  
3rd Year B.Tech CSE
