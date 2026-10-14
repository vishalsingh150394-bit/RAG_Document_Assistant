import os
import tempfile
import streamlit as st

# Streamlit Cloud secrets -> environment variables (used by doc_assistant.get_llm)
try:
    for k in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

import doc_assistant as da


def safe_md(text) -> str:
    """Escape ~ and $ so Streamlit markdown does not render strikethrough or LaTeX."""
    return str(text).replace("~", "\\~").replace("$", "\\$")


st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")
st.caption("LlamaIndex (indexing + retrieval) · LangChain (prompts, LLM, output parsing) · multi-document Q&A and summaries")

if "assistant" not in st.session_state:
    with st.spinner("Loading embedding model (first run takes a minute)..."):
        st.session_state.assistant = da.DocAssistant()
    st.session_state.messages = []      # [(role, text, sources)]
    st.session_state.done = set()
assistant = st.session_state.assistant

with st.sidebar:
    st.header("1. Add documents")
    files = st.file_uploader("PDF / TXT / MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if files:
        for f in files:
            key = (f.name, f.size)
            if key in st.session_state.done:
                continue
            with st.spinner(f"Indexing {f.name}..."):
                path = os.path.join(tempfile.mkdtemp(), f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                n = assistant.add_documents(da.load_files([path]))
            st.session_state.done.add(key)
            st.success(f"{f.name}: {n} chunks")
    url = st.text_input("...or a web page URL")
    if st.button("Add URL") and url.strip():
        try:
            with st.spinner("Fetching page..."):
                n = assistant.add_documents(da.load_web(url.strip()))
            st.success(f"Added page: {n} chunks")
        except Exception as e:
            st.error(f"Could not load URL: {e}")
    if st.button("Load sample documents"):
        paths = da.write_sample_docs(os.path.join(tempfile.mkdtemp(), "samples"))
        n = assistant.add_documents(da.load_files(paths))
        st.success(f"Sample documents added: {n} chunks")

    st.header("2. Search settings")
    docs = assistant.documents()
    chosen = st.multiselect("Search only in (empty = all documents)", list(docs))
    top_k = st.slider("Passages to retrieve (top-k)", 2, 10, 6)
    st.markdown("**Indexed documents**")
    for name, n in docs.items():
        st.write(f"• {name} ({n} chunks)")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

tab_chat, tab_sum = st.tabs(["💬 Ask questions", "📝 Summarize a document"])

with tab_chat:
    if not docs:
        st.info("Add documents in the sidebar (or click 'Load sample documents') to start.")
    for role, text, sources in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(safe_md(text))
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(safe_md(f"**[{s['id']}] {s['file']}** (page {s['page']}, score {s['score']})\n\n> {s['snippet']}"))
    q = st.chat_input("Ask something about your documents")
    if q:
        history = [(r, t) for r, t, _ in st.session_state.messages]
        st.session_state.messages.append(("user", q, None))
        with st.chat_message("user"):
            st.markdown(safe_md(q))
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                try:
                    res = assistant.ask(q, chosen or None, top_k, history)
                    text, src = res["answer"], (res["sources"] if res["found"] else [])
                except Exception as e:
                    text, src = f"Something went wrong: {e}", None
            st.markdown(safe_md(text))
            if src:
                with st.expander("Sources"):
                    for s in src:
                        st.markdown(safe_md(f"**[{s['id']}] {s['file']}** (page {s['page']}, score {s['score']})\n\n> {s['snippet']}"))
        st.session_state.messages.append(("assistant", text, src))

with tab_sum:
    if docs:
        name = st.selectbox("Document", list(docs))
        if st.button("Summarize"):
            with st.spinner("Summarising (map-reduce)..."):
                try:
                    st.markdown(safe_md(assistant.summarize(name)))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    else:
        st.info("Add a document first.")
