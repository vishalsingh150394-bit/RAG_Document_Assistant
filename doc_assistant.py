# %% [markdown]
# ## 1. Imports, LLM factory (LangChain) and embeddings (LlamaIndex)
# - **LlamaIndex**: document loading, chunking, vector index, retriever
# - **LangChain**: LLM integration, prompt templates, chains (LCEL), output parsing

# %%
import os
import re
import time
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from llama_index.core import (Document, VectorStoreIndex, SimpleDirectoryReader, Settings,
                              StorageContext, load_index_from_storage)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import MockLLM
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_llm(temperature: float = 0.1):
    """LangChain chat model chosen by env vars LLM_PROVIDER / LLM_MODEL."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "openai/gpt-oss-20b", temperature=temperature)   # GROQ_API_KEY
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-3.8-flash", temperature=temperature)  # GOOGLE_API_KEY
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)        # OPENAI_API_KEY
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-haiku-4-5-20251001", temperature=temperature)  # ANTHROPIC_API_KEY
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


_EMBED = None


def get_embed_model():
    """Free local embedding model (no API key). Loaded once."""
    global _EMBED
    if _EMBED is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _EMBED = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    return _EMBED


def setup_llamaindex(chunk_size: int = 512, chunk_overlap: int = 64):
    Settings.embed_model = get_embed_model()
    Settings.llm = MockLLM()          # LlamaIndex does retrieval only; LangChain generates the answers
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def run_with_retry(fn, tries: int = 4, wait: int = 20):
    """Retry temporary API errors (503 / 429 rate limits)."""
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            temp = any(x in msg for x in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "rate_limit", "overloaded"))
            if temp and i < tries:
                print(f"Temporary API error (attempt {i}/{tries}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# %% [markdown]
# ## 2. Ingestion: PDFs / text files / web pages -> LlamaIndex `Document`s

# %%
def _read_pdf(path: str) -> List[Document]:
    """One Document per page. Layout mode keeps each table row on one line ('label | v1 | v2')."""
    from pypdf import PdfReader
    name, docs = os.path.basename(path), []
    for i, page in enumerate(PdfReader(path).pages, 1):
        try:
            text = page.extract_text(extraction_mode="layout")
        except TypeError:                       # very old pypdf
            text = page.extract_text()
        lines = [re.sub(r"\s{2,}", " | ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
        if lines:
            docs.append(Document(text="\n".join(lines),
                                 metadata={"file_name": name, "page_label": str(i), "source_type": "pdf"}))
    return docs


def load_files(paths) -> List[Document]:
    """PDFs (page-wise, table-friendly) plus TXT / MD / DOCX via LlamaIndex SimpleDirectoryReader."""
    docs, others = [], []
    for p in paths:
        p = str(p)
        if p.lower().endswith(".pdf"):
            docs.extend(_read_pdf(p))
        else:
            others.append(p)
    if others:
        docs.extend(SimpleDirectoryReader(input_files=others).load_data())
    return docs


def load_web(url: str) -> List[Document]:
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)[:80]
    return [Document(text=text, metadata={"file_name": title, "source_type": "web", "url": url})]


SAMPLE_DOCS = {
    "rolling_forecast_guide.txt": """Acme Retail Rolling Forecast Guide.
Acme maintains a 12-month rolling forecast. After every month-end close the forecast is extended by one month, so the horizon always stays at twelve months. The forecast is refreshed on the fifth working day of each month.
The forecast is driver based: revenue equals orders multiplied by average order value (AOV). Sales volume assumptions come from the Category team, marketing spend from the Marketing team, and fulfilment cost per order from the Operations team.
Accuracy targets: total company revenue within 3% of actuals and category level revenue within 5% of actuals.
Version control: every refresh is saved with a version tag such as v-202610. If a refresh moves forecast net revenue by more than 2%, the Finance Director must approve the change before it is published.""",
    "variance_analysis_policy.txt": """Acme Retail Variance Analysis Policy.
Variance is calculated as actual minus budget. A variance is favourable when it increases profit and unfavourable when it reduces profit.
Investigation threshold: any line item that deviates by more than 5% from budget and by more than INR 10 lakh must be investigated.
Variances are split into volume, price and mix effects so that the root cause is clear.
Every investigated variance needs written commentary stating the cause, the financial impact and the action owner. Commentary is due within three working days of the month-end close.
If the same line item shows an unfavourable variance for three months in a row, a re-forecast of that line is triggered.""",
    "ecommerce_kpi_handbook.txt": """Acme Retail E-commerce KPI Handbook.
Average order value (AOV) equals net revenue divided by the number of orders.
Conversion rate equals orders divided by sessions. The internal benchmark is 2.5%.
Customer acquisition cost (CAC) equals marketing spend divided by new customers. The target is below INR 450.
Gross margin equals net revenue minus cost of goods sold, divided by net revenue. Contribution margin also deducts fulfilment, payment gateway and marketing costs; the target is 8%.
The KPI review meeting is held every Monday. The Power BI dashboards are refreshed daily at 6 AM.""",
}


def write_sample_docs(folder: str = "sample_docs") -> List[str]:
    os.makedirs(folder, exist_ok=True)
    paths = []
    for name, text in SAMPLE_DOCS.items():
        p = os.path.join(folder, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(p)
    return paths


# %% [markdown]
# ## 3. LangChain prompts, chains and output parsing
# - `ChatPromptTemplate` for question condensing, answering, map and reduce summaries
# - `PydanticOutputParser` turns the LLM text into a typed `DocAnswer`; if parsing fails we fall back to plain text

# %%
class DocAnswer(BaseModel):
    answer: str = Field(description="Answer using only the context passages. Mention passage numbers like [1].")
    cited_sources: List[int] = Field(description="Numbers of the context passages actually used, e.g. [1, 3]")
    found_in_documents: bool = Field(description="False if the context does not contain the answer")


answer_parser = PydanticOutputParser(pydantic_object=DocAnswer)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful document assistant. Answer ONLY from the numbered context passages. "
     "If the answer is not in the context, say you could not find it in the documents and set found_in_documents to false. "
     "Never use outside knowledge. Text from tables appears as rows like 'label | value | value': match each label only with the values on its own row, and check the column order. "
     "Be concise and mention passage numbers like [1].\n\n{format_instructions}"),
    ("human", "Conversation so far:\n{history}\n\nContext passages:\n{context}\n\nQuestion: {question}"),
]).partial(format_instructions=answer_parser.get_format_instructions())

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the follow-up question as a standalone question using the conversation. "
               "Return only the rewritten question."),
    ("human", "Conversation:\n{history}\n\nFollow-up question: {question}"),
])

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarise the passage in 2-3 sentences. Keep facts and numbers."),
    ("human", "{text}"),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Combine the partial summaries of one document into a clear summary: "
               "one overview sentence followed by 4-6 bullet points with the key facts and numbers."),
    ("human", "Document: {name}\n\nPartial summaries:\n{parts}"),
])


def _safe_parse(text: str) -> DocAnswer:
    try:
        return answer_parser.parse(text)
    except Exception:
        return DocAnswer(answer=text.strip(), cited_sources=[], found_in_documents=True)


def build_chains():
    llm = get_llm()
    return {
        "qa": QA_PROMPT | llm | StrOutputParser() | RunnableLambda(_safe_parse),
        "condense": CONDENSE_PROMPT | llm | StrOutputParser(),
        "map": MAP_PROMPT | llm | StrOutputParser(),
        "reduce": REDUCE_PROMPT | llm | StrOutputParser(),
    }


# %% [markdown]
# ## 4. The assistant: LlamaIndex index + retriever, LangChain generation
# Features: multi-document search, per-document filter (metadata filters), conversational follow-ups, source citations, per-document summaries, save / load index.

# %%
class DocAssistant:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        setup_llamaindex(chunk_size, chunk_overlap)
        self.index = VectorStoreIndex(nodes=[])
        self.chunks = {}                      # file_name -> number of chunks
        self._chains = None

    @property
    def chains(self):
        if self._chains is None:
            self._chains = build_chains()
        return self._chains

    # ---- indexing (LlamaIndex) ----
    def add_documents(self, docs: List[Document]) -> int:
        nodes = Settings.node_parser.get_nodes_from_documents(docs)
        self.index.insert_nodes(nodes)
        for n in nodes:
            name = n.metadata.get("file_name", "unknown")
            self.chunks[name] = self.chunks.get(name, 0) + 1
        return len(nodes)

    def documents(self) -> dict:
        return dict(self.chunks)

    def save(self, path: str):
        self.index.storage_context.persist(persist_dir=path)

    def load(self, path: str):
        self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=path))
        self.chunks = {}
        for n in self.index.docstore.docs.values():
            name = n.metadata.get("file_name", "unknown")
            self.chunks[name] = self.chunks.get(name, 0) + 1

    # ---- retrieval (LlamaIndex retriever) ----
    def retrieve(self, question: str, doc_names: Optional[List[str]] = None, k: int = 4):
        filters = None
        if doc_names:
            filters = MetadataFilters(filters=[MetadataFilter(key="file_name", value=list(doc_names),
                                                               operator=FilterOperator.IN)])
        return self.index.as_retriever(similarity_top_k=k, filters=filters).retrieve(question)

    # ---- generation (LangChain) ----
    def ask(self, question: str, doc_names: Optional[List[str]] = None, k: int = 4, history=None) -> dict:
        if not self.chunks:
            return {"answer": "No documents are indexed yet. Upload a file or add a URL first.",
                    "found": False, "sources": [], "all_sources": [], "standalone_question": question}
        history = (history or [])[-6:]
        hist_txt = "\n".join(f"{role}: {text}" for role, text in history) or "(none)"
        standalone = question
        if history:
            standalone = run_with_retry(lambda: self.chains["condense"].invoke(
                {"history": hist_txt, "question": question})).strip() or question
        nodes = self.retrieve(standalone, doc_names, k)
        context = "\n\n".join(
            f"[{i}] (source: {n.metadata.get('file_name')}, page {n.metadata.get('page_label', '-')}) {n.get_content()}"
            for i, n in enumerate(nodes, 1))
        res = run_with_retry(lambda: self.chains["qa"].invoke(
            {"history": hist_txt, "context": context, "question": standalone}))
        if not res.found_in_documents:           # small models sometimes miss: ask once more before giving up
            retry = run_with_retry(lambda: self.chains["qa"].invoke(
                {"history": hist_txt, "context": context, "question": standalone}))
            if retry.found_in_documents:
                res = retry
        all_sources = [{"id": i, "file": n.metadata.get("file_name"), "page": n.metadata.get("page_label", "-"),
                        "score": round(float(n.score), 3) if n.score is not None else None,
                        "snippet": n.get_content()[:300]} for i, n in enumerate(nodes, 1)]
        cited = [s for s in all_sources if s["id"] in res.cited_sources] or all_sources
        return {"answer": res.answer, "found": res.found_in_documents, "sources": cited,
                "all_sources": all_sources, "standalone_question": standalone}

    def summarize(self, doc_name: str, max_chunks: int = 8) -> str:
        nodes = [n for n in self.index.docstore.docs.values() if n.metadata.get("file_name") == doc_name]
        if not nodes:
            return f"Document '{doc_name}' not found."
        if len(nodes) > max_chunks:               # sample evenly to stay within API limits
            step = len(nodes) / max_chunks
            nodes = [nodes[int(i * step)] for i in range(max_chunks)]
        parts = run_with_retry(lambda: self.chains["map"].batch(
            [{"text": n.get_content()} for n in nodes], config={"max_concurrency": 2}))
        joined = "\n".join(f"- {p.strip()}" for p in parts)
        return run_with_retry(lambda: self.chains["reduce"].invoke({"name": doc_name, "parts": joined}))
