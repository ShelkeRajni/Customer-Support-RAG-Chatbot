import os
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_documents
from src.chunking import split_documents
from src.embedding import get_embeddings_model
from src.vector_store import build_vector_store, save_vector_store, load_vector_store
from src.retriever import get_retriever
from src.chatbot import get_chatbot_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="SupportGPT - Customer Support RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .gradient-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .source-card {
        background-color: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 8px;
        padding: 12px;
        margin: 5px 0px;
        font-size: 0.85rem;
    }
    .source-header { font-weight: 600; color: #6366f1; margin-bottom: 4px; }
    .source-content { font-style: italic; }
    </style>
""", unsafe_allow_html=True)

DOCS_DIR = os.path.join("data", "company_docs")
INDEX_DIR = "faiss_index"
os.makedirs(DOCS_DIR, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712038.png", width=80)
    st.markdown("## SupportGPT")
    st.markdown("---")

    st.markdown("### 🔑 Gemini API Key")
    existing_key = os.getenv("GOOGLE_API_KEY", "")
    user_api_key = st.text_input(
        "Enter your free Gemini API Key",
        value=existing_key,
        type="password",
        placeholder="AIza...",
    )
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
    st.caption("🆓 Get free key: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)")

    st.markdown("---")
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Ingest Documents"):
            if not os.environ.get("GOOGLE_API_KEY"):
                st.error("Please enter your Gemini API Key first!")
            else:
                with st.spinner("Ingesting documents via Gemini API..."):
                    try:
                        for f in os.listdir(DOCS_DIR):
                            try:
                                os.remove(os.path.join(DOCS_DIR, f))
                            except Exception:
                                pass
                        for file in uploaded_files:
                            try:
                                with open(os.path.join(DOCS_DIR, file.name), "wb") as f:
                                    f.write(file.getbuffer())
                            except Exception as e:
                                st.warning(f"Could not save {file.name}: {e}")
                        docs = load_documents(DOCS_DIR)
                        if docs:
                            chunks = split_documents(docs)
                            embeddings = get_embeddings_model()
                            vs = build_vector_store(chunks, embeddings)
                            save_vector_store(vs, INDEX_DIR)
                            st.session_state.vector_store = vs
                            st.success(f"Done! Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
                        else:
                            st.error("No readable text found in the uploaded files.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📊 System Status")
    docs_list = os.listdir(DOCS_DIR)
    if docs_list:
        st.info("📁 Documents:\n" + "\n".join([f"- {d}" for d in docs_list]))
    else:
        st.warning("No documents uploaded yet.")

    if st.session_state.vector_store is None:
        idx_path = os.path.join(INDEX_DIR, "index.faiss")
        if os.path.exists(idx_path) and os.environ.get("GOOGLE_API_KEY"):
            try:
                embeddings = get_embeddings_model()
                st.session_state.vector_store = load_vector_store(INDEX_DIR, embeddings)
                st.success("Loaded saved index!")
            except Exception as e:
                st.warning(f"Could not load saved index: {e}")

    if st.session_state.vector_store:
        st.success("🟢 Vector DB: Ready")
    else:
        st.error("🔴 Vector DB: Not loaded")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
    <div style="display:flex; align-items:center; gap:15px; margin-bottom:20px;">
        <span style="font-size:3rem;">🤖</span>
        <div>
            <h1 class="gradient-title" style="margin:0;">SupportGPT</h1>
            <p style="margin:0; color:#6b7280; font-size:1.1rem;">
                Embeddings + Chat: Google Gemini API &nbsp;|&nbsp; Vector DB: FAISS
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

if not st.session_state.vector_store:
    st.info(
        "👋 **Welcome to SupportGPT!**\n\n"
        "**Step 1:** Paste your **Gemini API Key** in the sidebar\n\n"
        "**Step 2:** Upload a **PDF or TXT** document\n\n"
        "**Step 3:** Click **🚀 Process & Ingest Documents**\n\n"
        "**Step 4:** Once status shows **🟢 Vector DB: Ready** — start chatting!"
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 View Sources"):
                    for idx, doc in enumerate(msg["sources"]):
                        src = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", None)
                        page_str = f" · Page {page + 1}" if page is not None else ""
                        st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">📄 {os.path.basename(src)}{page_str}</div>
                                <div class="source-content">"{doc.page_content[:300]}..."</div>
                            </div>
                        """, unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question about your documents..."):
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please enter your Gemini API Key in the sidebar!")
        else:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("*Thinking...*")
                try:
                    retriever = get_retriever(st.session_state.vector_store, k=3)
                    chain = get_chatbot_chain(retriever, llm_provider="gemini")
                    chat_history = []
                    for m in st.session_state.messages[:-1]:
                        if m["role"] == "user":
                            chat_history.append(HumanMessage(content=m["content"]))
                        elif m["role"] == "assistant":
                            chat_history.append(AIMessage(content=m["content"]))
                    response = chain.invoke({"input": prompt, "chat_history": chat_history})
                    answer = response["answer"]
                    sources = response.get("context", [])
                    placeholder.markdown(answer)
                    if sources:
                        with st.expander("📚 View Sources"):
                            for idx, doc in enumerate(sources):
                                src = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", None)
                                page_str = f" · Page {page + 1}" if page is not None else ""
                                st.markdown(f"""
                                    <div class="source-card">
                                        <div class="source-header">📄 {os.path.basename(src)}{page_str}</div>
                                        <div class="source-content">"{doc.page_content[:300]}..."</div>
                                    </div>
                                """, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    placeholder.error(f"Error: {e}")

