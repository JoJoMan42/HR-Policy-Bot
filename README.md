# 🧠 HR Policy RAG Agent

An intelligent HR assistant built using **Retrieval-Augmented Generation (RAG)** that answers employee queries based strictly on company HR policies.

---

## 📸 Demo / Preview

<!-- Replace with your screenshot file -->

![App Screenshot](https://github.com/JoJoMan42/HR-Policy-Bot/blob/master/picture.png)

---

## 🚀 Features

* 📄 Answers HR-related questions using company policy documents
* 🔍 Uses vector search (ChromaDB) for accurate context retrieval
* 🤖 Generates grounded responses using an LLM
* 🛠️ Supports tools:

  * Leave balance calculator
  * Current date/time queries
* 📊 Evaluated using RAGAS metrics (answer relevance, context precision)

---

## 🧠 How It Works

1. HR policy PDF is loaded and split into chunks
2. Text is converted into embeddings
3. Stored in a vector database (ChromaDB)
4. User query → converted to embedding
5. Relevant chunks retrieved
6. LLM generates answer using retrieved context
7. Tools are invoked when needed (e.g., leave calculation)

---

## 🏗️ Tech Stack

* Python
* ChromaDB (Vector Database)
* LLM APIs (OpenAI / Groq)
* RAGAS (Evaluation)
* Streamlit (Frontend)

---

## 📂 Project Structure

```
agent.py                  # Core agent logic
capstone_streamlit.py     # Streamlit UI
day13_capstone.ipynb      # Development notebook
hr_policy.pdf             # Knowledge base
requirements.txt          # Dependencies (optional)
README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/hr-policy-rag-agent.git
cd hr-policy-rag-agent
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```
streamlit run capstone_streamlit.py
```

---

## 📊 Evaluation

The system is evaluated using **RAGAS metrics**:

* Answer Relevance
* Context Precision
* Faithfulness

---

## ⚠️ Limitations

* Performance depends on quality of HR documents
* Requires API key for LLM usage
* Retrieval may fail for vague queries
* Due to free-tier Groq, proper RAGAS Evaluation may not happen

---

## 💡 Future Improvements

* Better retrieval (reranking / hybrid search)
* Multi-document support
* Improved conversation memory
* Cloud deployment

