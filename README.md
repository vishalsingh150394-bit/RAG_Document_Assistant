# RAG-Based Intelligent Document Assistant (LangChain + LlamaIndex)

Upload PDFs / text files or add a web page, then ask questions with cited sources, search a single document or all of them, and get per-document summaries.

- **LlamaIndex:** loading (`SimpleDirectoryReader`), chunking (`SentenceSplitter`), `VectorStoreIndex`, retriever with metadata filters, index persistence
- **LangChain:** LLM integration (Groq / Gemini / OpenAI / Anthropic), prompt templates, LCEL chains, `PydanticOutputParser`, map-reduce summaries
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free)

## Run locally
```bash
pip install -r requirements.txt
export LLM_PROVIDER=groq GROQ_API_KEY=your_key LLM_MODEL=openai/gpt-oss-20b
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
1. Push these files to a public GitHub repo (`streamlit_app.py`, `doc_assistant.py`, `requirements.txt`, `README.md`, `.streamlit/config.toml`).
2. share.streamlit.io -> Create app -> main file `streamlit_app.py` -> Advanced settings: **Python 3.12**, and Secrets:
```
LLM_PROVIDER = "groq"
LLM_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = "your_key"
```
