import streamlit as st
import uuid
from agent import (
    load_embedder,
    load_llm,
    load_documents_from_pdf,
    build_chromadb,
    HRAgent,
    PDF_PATH
)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Tyrell Corp HR Assistant",
    page_icon  = "🤖",
    layout     = "wide"
)

# ──────────────────────────────────────────────
# CACHED INITIALISATION
# Everything built once, reused on every rerun.
# ──────────────────────────────────────────────
@st.cache_resource
def initialise():
    embedder   = load_embedder()
    llm        = load_llm()
    documents  = load_documents_from_pdf(PDF_PATH)
    collection = build_chromadb(documents, embedder)
    agent      = HRAgent(llm, embedder, collection)
    return agent

agent = initialise()

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "messages"  not in st.session_state:
    st.session_state.messages  = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/robot.png", width=80)
    st.title("HR Assistant")
    st.caption("Powered by LangGraph + Groq")

    st.divider()

    st.markdown("### 🏢 About")
    st.markdown(
        "This is the **Tyrell Corp HR Policy Assistant**. "
        "Ask me anything about company policies and I'll answer "
        "from the official HR handbook."
    )

    st.divider()

    st.markdown("### 📋 Topics I can help with")
    topics = [
        "🏖️  Leave Policy",
        "🏠  Work From Home",
        "🕘  Working Hours",
        "💰  Salary & Payroll",
        "📝  Notice Period",
        "🧾  Reimbursements",
        "📅  Public Holidays",
        "⚖️  Code of Conduct",
        "⚠️  Disciplinary Policy",
        "🏥  Health & Insurance",
    ]
    for topic in topics:
        st.markdown(f"- {topic}")

    st.divider()

    st.markdown("### 🔑 Session Info")
    st.code(f"Thread: {st.session_state.thread_id[:8]}...", language=None)
    if st.session_state.user_name:
        st.success(f"👤 {st.session_state.user_name}")

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.user_name = None
        st.rerun()

# ──────────────────────────────────────────────
# MAIN CHAT
# ──────────────────────────────────────────────
st.title("🤖 Tyrell Corp HR Policy Assistant")
st.caption(
    "Ask me about leave, salary, WFH, notice period, benefits, and more. "
    "I only answer from the official HR handbook — no guessing."
)

st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            with st.expander("📊 Response Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Route",        meta.get("route", "N/A"))
                col2.metric("Faithfulness", f"{meta.get('faithfulness', 0):.2f}")
                col3.metric("Sources",      len(meta.get("sources", [])))
                if meta.get("sources"):
                    st.markdown("**Retrieved from:**")
                    for src in meta["sources"]:
                        st.markdown(f"- 📄 {src}")

# Chat input
if prompt := st.chat_input("Ask an HR question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking up HR policy..."):
            result = agent.ask(prompt, thread_id=st.session_state.thread_id)

        answer       = result.get("answer", "Sorry, I could not generate a response.")
        route        = result.get("route", "N/A")
        faithfulness = result.get("faithfulness", 0.0)
        sources      = result.get("sources", [])
        user_name    = result.get("user_name", None)

        if user_name:
            st.session_state.user_name = user_name

        st.markdown(answer)

        with st.expander("📊 Response Details", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Route",        route)
            col2.metric("Faithfulness", f"{faithfulness:.2f}")
            col3.metric("Sources",      len(sources))
            if sources:
                st.markdown("**Retrieved from:**")
                for src in sources:
                    st.markdown(f"- 📄 {src}")

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "meta"   : {
            "route"       : route,
            "faithfulness": faithfulness,
            "sources"     : sources
        }
    })

# Empty state
if not st.session_state.messages:
    st.markdown("### 👋 Welcome! Here are some questions to get started:")
    starter_questions = [
        "How many paid leaves do I get per year?",
        "What is the notice period if I want to resign?",
        "Can I work from home?",
        "When is my salary credited?",
        "What health insurance benefits do I have?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(starter_questions):
        with cols[i % 2]:
            st.info(f"💬 {q}")