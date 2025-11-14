# test.py (updated) — Safe startup + deferred heavy imports + logging
# Purpose: Prevent crashes at import-time by deferring heavy work and logging full tracebacks.
# Usage: streamlit run test.py

import os
import sys
import time
import logging
import traceback
from datetime import datetime
from typing import List, Optional, Tuple, Any

# --------- Logging setup (file) ----------
LOGFILE = os.environ.get("RAG_STARTUP_LOG", "startup_error.log")
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE,
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
)

# Lightweight imports safe at module import time
try:
    import streamlit as st
    st.set_page_config(page_title="DocQA — Safe RAG", layout="wide")
except Exception:
    tb = traceback.format_exc()
    logging.error("Streamlit import failed:\n" + tb)
    print("Streamlit import failed. See startup_error.log for traceback.", file=sys.stderr)
    raise

# Show startup errors (if any) in the UI sidebar so you can inspect them quickly
def show_startup_log_in_sidebar():
    if os.path.exists(LOGFILE):
        try:
            with open(LOGFILE, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if txt.strip():
                st.sidebar.markdown("### Startup / Import Log")
                st.sidebar.text_area("startup_error.log", value=txt, height=250)
        except Exception:
            pass

show_startup_log_in_sidebar()

# --------- Defer heavy imports into functions ---------
# Each function will attempt to import the module the first time it's needed.
_heavy = {}  # cache for imported modules


def import_genai():
    """Return genai client module (imported). Raises informative error on failure."""
    if "genai" in _heavy:
        return _heavy["genai"]
    try:
        from google import genai  # type: ignore
        _heavy["genai"] = genai
        logging.info("Imported google-genai successfully.")
        return genai
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("google-genai import failed:\n" + tb)
        raise RuntimeError("google-genai import failed: " + str(e))


def import_faiss():
    if "faiss" in _heavy:
        return _heavy["faiss"]
    try:
        import faiss
        _heavy["faiss"] = faiss
        logging.info("Imported faiss successfully.")
        return faiss
    except Exception as e:
        tb = traceback.format_exc()
        logging.warning("faiss import failed (will use numpy fallback):\n" + tb)
        _heavy["faiss"] = None
        return None


def import_sentence_transformers():
    if "st" in _heavy:
        return _heavy["st"]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _heavy["st"] = SentenceTransformer
        logging.info("Imported sentence-transformers successfully.")
        return SentenceTransformer
    except Exception as e:
        tb = traceback.format_exc()
        logging.warning("sentence-transformers import failed:\n" + tb)
        _heavy["st"] = None
        return None


def import_pdfplumber():
    if "pdfplumber" in _heavy:
        return _heavy["pdfplumber"]
    try:
        import pdfplumber
        _heavy["pdfplumber"] = pdfplumber
        return pdfplumber
    except Exception:
        _heavy["pdfplumber"] = None
        return None


def import_docx():
    if "docx" in _heavy:
        return _heavy["docx"]
    try:
        import docx
        _heavy["docx"] = docx
        return docx
    except Exception:
        _heavy["docx"] = None
        return None


def import_pandas():
    if "pd" in _heavy:
        return _heavy["pd"]
    try:
        import pandas as pd
        _heavy["pd"] = pd
        return pd
    except Exception:
        _heavy["pd"] = None
        return None


def import_bs4():
    if "bs4" in _heavy:
        return _heavy["bs4"]
    try:
        from bs4 import BeautifulSoup  # type: ignore
        _heavy["bs4"] = BeautifulSoup
        return BeautifulSoup
    except Exception:
        _heavy["bs4"] = None
        return None


def import_tiktoken():
    if "tiktoken" in _heavy:
        return _heavy["tiktoken"]
    try:
        import tiktoken
        _heavy["tiktoken"] = tiktoken
        return tiktoken
    except Exception:
        _heavy["tiktoken"] = None
        return None


def import_numpy():
    if "np" in _heavy:
        return _heavy["np"]
    try:
        import numpy as np
        _heavy["np"] = np
        return np
    except Exception:
        tb = traceback.format_exc()
        logging.error("numpy import failed:\n" + tb)
        raise


def import_sklearn_tfidf():
    if "sklearn" in _heavy:
        return _heavy["sklearn"]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        _heavy["sklearn"] = (TfidfVectorizer, cosine_similarity)
        return _heavy["sklearn"]
    except Exception:
        _heavy["sklearn"] = None
        return None

# --------- Simple utilities (safe) ----------
def safe_write_log(msg: str):
    try:
        logging.info(msg)
    except Exception:
        pass


# --------- Text extraction functions (deferred imports inside) ----------
def extract_text_from_pdf_bytes(b: bytes, enable_ocr: bool = False) -> str:
    pdfplumber = import_pdfplumber()
    if pdfplumber is None:
        return ""
    parts = []
    try:
        import io
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t.strip())
    except Exception as e:
        logging.warning("PDF extraction error: " + str(e) + "\n" + traceback.format_exc())
    return "\n\n".join([p for p in parts if p])


def extract_text_from_docx_bytes(b: bytes) -> str:
    docx = import_docx()
    if docx is None:
        return ""
    try:
        import io
        d = docx.Document(io.BytesIO(b))
        paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paras)
    except Exception as e:
        logging.warning("DOCX extraction error: " + str(e) + "\n" + traceback.format_exc())
        return ""


def extract_text_from_excel_bytes(b: bytes) -> str:
    pd = import_pandas()
    if pd is None:
        return ""
    try:
        import io
        df = pd.read_excel(io.BytesIO(b), engine="openpyxl")
        rows = []
        for _, r in df.iterrows():
            rows.append(" | ".join([str(x) for x in r.tolist() if pd.notna(x)]))
        return "\n".join(rows)
    except Exception:
        try:
            import io
            df = pd.read_csv(io.BytesIO(b))
            rows = []
            for _, r in df.iterrows():
                rows.append(" | ".join([str(x) for x in r.tolist() if pd.notna(x)]))
            return "\n".join(rows)
        except Exception as e:
            logging.warning("Excel extraction failed: " + str(e))
            return ""


def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return str(b)


def fetch_webpage_text(url: str) -> str:
    BeautifulSoup = import_bs4()
    if BeautifulSoup is None:
        return ""
    try:
        import requests
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        if soup.body:
            return soup.body.get_text(separator="\n")
        return soup.get_text(separator="\n")
    except Exception as e:
        logging.warning("Web fetch failed: " + str(e))
        return ""


def simple_clean(text: str) -> str:
    return "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])


# --------- Chunking (token-aware fallback) ----------
def chunk_text_char_based(text: str, size: int = 1600, overlap: int = 400) -> List[str]:
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) > size:
                for k in range(0, len(p), size - overlap):
                    chunks.append(p[k:k+size].strip())
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    if overlap and len(chunks) > 1:
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                prev = out[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                out.append((tail + "\n\n" + c).strip())
        chunks = out
    return chunks


def chunk_text_tokenwise(text: str, max_tokens: int = 400, overlap_tokens: int = 80, tokenizer_name: str = "cl100k_base") -> List[str]:
    if not text:
        return []
    tiktoken = import_tiktoken()
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding(tokenizer_name)
            tokens = enc.encode(text)
            chunks = []
            i = 0
            L = len(tokens)
            while i < L:
                j = min(i + max_tokens, L)
                chunk_tokens = tokens[i:j]
                chunk_text = enc.decode(chunk_tokens)
                chunks.append(chunk_text)
                i = j - overlap_tokens
                if i < 0:
                    i = 0
            return chunks
        except Exception:
            pass
    char_size = max(400, max_tokens * 4)
    char_overlap = max(50, overlap_tokens * 4)
    return chunk_text_char_based(text, size=char_size, overlap=char_overlap)


# --------- GenAI client & embedding helpers (deferred) ----------
def init_genai_client_safe(api_key: str):
    """
    Initialize google-genai Client lazily when the user clicks Init.
    Returns client instance or raises informative error.
    """
    try:
        genai = import_genai()
        client = genai.Client(api_key=api_key)
        safe_write_log("GenAI client initialized")
        return client
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("GenAI init failed:\n" + tb)
        raise RuntimeError("GenAI init failed: " + str(e))


def embed_texts_gemini_with_logging(client, texts: List[str], model: str = "gemini-embedding-001", max_retries: int = 1):
    """
    Request embeddings from Gemini. Returns normalized numpy array (n,d) or raises.
    Logs full traceback to logfile on failure.
    """
    np = import_numpy()
    if client is None:
        raise RuntimeError("GenAI client not initialized.")
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    attempt = 0
    while attempt <= max_retries:
        try:
            attempt += 1
            safe_write_log(f"Requesting embeddings (attempt {attempt}) model={model} count={len(texts)}")
            resp = client.models.embed_content(model=model, contents=texts)
            # robust extraction
            try:
                embs = np.array([e.embedding for e in resp.embeddings], dtype="float32")
            except Exception:
                if isinstance(resp, dict) and "embeddings" in resp:
                    embs = np.array([e["embedding"] for e in resp["embeddings"]], dtype="float32")
                else:
                    raise RuntimeError("Unexpected embedding response structure: " + str(type(resp)))
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = (embs / norms).astype("float32")
            safe_write_log("Embeddings created successfully.")
            return embs
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Embedding attempt {attempt} failed:\n" + tb)
            if attempt <= max_retries:
                time.sleep(1.0)
                continue
            else:
                raise


# --------- Retrieval helpers ----------
def build_faiss_index(embs):
    faiss = import_faiss()
    np = import_numpy()
    if faiss is None:
        raise RuntimeError("faiss not available")
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embs)
    idx.add(embs)
    return idx


def nearest_neighbors_search(query_vec, all_embs, top_k=10):
    import numpy as np
    # both normalized; use dot product
    sims = (all_embs @ query_vec.T).squeeze()
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idxs]


def build_combined_context(chunks: List[str], metas: List[dict], selected_idxs: List[int], max_chars=8000):
    """
    Combine selected chunks into a single context string, respecting a char budget.
    """
    pieces = []
    used = []
    current = 0
    for idx in selected_idxs:
        text = chunks[idx]
        src = metas[idx].get("source", "unknown")
        candidate = f"[{src}] {text}"
        if current + len(candidate) > max_chars:
            # try adding a snippet
            snippet = text[:1500]
            if current + len(snippet) <= max_chars:
                pieces.append(f"[{src}] {snippet}")
                used.append((idx, snippet))
            break
        pieces.append(candidate)
        used.append((idx, text))
        current += len(candidate)
    return "\n\n---\n\n".join(pieces), used


# --------- LLM generation wrapper (Gemini) ----------
def generate_answer_gemini(client, model: str, question: str, context: str, max_output_tokens: int = 512):
    """
    Ask Gemini to generate an answer using the provided context.
    """
    if client is None:
        raise RuntimeError("GenAI client not initialized.")
    prompt = (
        "You are a helpful assistant. Use ONLY the CONTEXT below to answer the QUESTION. "
        "If the answer is not present, say 'I don't know'. Cite sources in square brackets.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely and cite sources."
    )
    resp = client.models.generate_content(model=model, contents=prompt)
    # robustly extract text
    try:
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception:
        return str(resp)


# --------- Session-state initialization (Streamlit) ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "metas" not in st.session_state:
    st.session_state.metas = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "backend" not in st.session_state:
    st.session_state.backend = None
if "client" not in st.session_state:
    st.session_state.client = None


# --------- Sidebar controls ----------
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("Google AI Studio API key (optional)", type="password")
init_button = st.sidebar.button("Init GenAI client")
embed_model_input = st.sidebar.text_input("Embedding model", value="gemini-embedding-001")
gen_model_input = st.sidebar.text_input("Generation model (default)", value="gemini-2.5-flash-lite-preview-06-17")
chunk_token_size = st.sidebar.number_input("Chunk token size (approx)", min_value=128, max_value=2000, value=600, step=50)
overlap_tokens = st.sidebar.number_input("Overlap tokens", min_value=0, max_value=400, value=80, step=10)
top_k = st.sidebar.slider("Top K retrieved chunks", min_value=3, max_value=30, value=15)
max_context_chars = st.sidebar.number_input("Max context size (chars)", min_value=1000, max_value=20000, value=8000, step=500)
enable_ocr = st.sidebar.checkbox("Enable OCR for scanned PDFs", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("Startup log (for debugging) below. If the app crashed during startup, the traceback will appear here.")
show_startup_log_in_sidebar()


# --------- Init client handler (deferred) ----------
def handle_init_client():
    key = api_key_input.strip() or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY")
    if not key:
        st.sidebar.error("No API key provided. Set GOOGLE_API_KEY env or paste in the sidebar.")
        return
    try:
        client = init_genai_client_safe(key)
        st.session_state.client = client
        st.sidebar.success("GenAI client initialized.")
    except Exception as e:
        st.sidebar.error(f"GenAI init failed: {e}")
        # log traceback to file
        tb = traceback.format_exc()
        logging.error("GenAI init handler exception:\n" + tb)
        show_startup_log_in_sidebar()


if init_button:
    handle_init_client()


# --------- Main UI: Upload & Ingest ----------
st.title("📚 DocQA — Safe Retrieval-Augmented Generation")
st.header("1) Upload documents and build index")

uploaded_files = st.file_uploader("Upload PDF/DOCX/XLSX/CSV/TXT (multiple)", accept_multiple_files=True, type=["pdf", "docx", "xlsx", "csv", "txt"])
url_input = st.text_input("Or provide a web / Confluence URL (optional)")

if st.button("Ingest & Build Index"):
    all_texts = []
    all_metas = []
    if not uploaded_files and not url_input:
        st.error("Upload files or provide a URL.")
    else:
        st.info("Extracting text from uploaded files...")
        for up in uploaded_files or []:
            try:
                name = up.name
                b = up.read()
                ext = name.lower().split(".")[-1]
                txt = ""
                if ext == "pdf":
                    txt = extract_text_from_pdf_bytes(b, enable_ocr=enable_ocr)
                elif ext == "docx":
                    txt = extract_text_from_docx_bytes(b)
                elif ext in ("xlsx", "xls"):
                    txt = extract_text_from_excel_bytes(b)
                elif ext in ("csv", "txt"):
                    txt = extract_text_from_txt_bytes(b)
                else:
                    txt = extract_text_from_txt_bytes(b)
                txt = simple_clean(txt)
                if not txt:
                    st.warning(f"No text found in {name}")
                    continue
                chunks = chunk_text_tokenwise(txt, max_tokens=chunk_token_size, overlap_tokens=overlap_tokens)
                for i, c in enumerate(chunks):
                    all_texts.append(c)
                    all_metas.append({"source": name, "chunk_id": i, "ingested_at": datetime.utcnow().isoformat()})
            except Exception as e:
                st.warning(f"Failed to process {up.name}: {e}\nSee startup log for details.")
                logging.error(f"Failed processing file {up.name}:\n" + traceback.format_exc())

        if url_input:
            st.info("Fetching web content...")
            page = fetch_webpage_text(url_input)
            page = simple_clean(page)
            if page:
                chunks = chunk_text_tokenwise(page, max_tokens=chunk_token_size, overlap_tokens=overlap_tokens)
                for i, c in enumerate(chunks):
                    all_texts.append(c)
                    all_metas.append({"source": url_input, "chunk_id": i, "ingested_at": datetime.utcnow().isoformat()})

        if not all_texts:
            st.error("No text ingested.")
        else:
            st.success(f"Ingested {len(all_texts)} chunks. Building embeddings (if GenAI client exists)...")
            # Attempt embeddings with Gemini if client initialized, else TF-IDF fallback
            try:
                client = st.session_state.client
                if client:
                    try:
                        embs = embed_texts_gemini_with_logging(client, all_texts, model=embed_model_input)
                        st.session_state.embeddings = embs
                        st.session_state.backend = "gemini"
                        # build FAISS if available
                        faiss_mod = import_faiss()
                        if faiss_mod:
                            try:
                                idx = build_faiss_index(embs)
                                st.session_state.faiss_index = idx
                                st.success("FAISS index built.")
                            except Exception as e:
                                st.warning(f"FAISS index build failed: {e}")
                                st.session_state.faiss_index = None
                        else:
                            st.info("FAISS not available; using numpy fallback for similarity search.")
                    except Exception as e:
                        st.warning(f"Embeddings failed: {e}\nFalling back to TF-IDF.")
                        logging.error("Embedding failure during ingest:\n" + traceback.format_exc())
                        # fall back to TF-IDF below
                        st.session_state.embeddings = None
                        st.session_state.backend = None

                if st.session_state.embeddings is None:
                    # TF-IDF fallback
                    st.info("Building TF-IDF index (fallback).")
                    tf_res = import_sklearn_tfidf()
                    if tf_res is None:
                        st.error("scikit-learn not installed; cannot build TF-IDF fallback.")
                    else:
                        TfidfVectorizer, cosine_similarity = tf_res
                        vec = TfidfVectorizer(max_features=65536, stop_words="english")
                        mat = vec.fit_transform(all_texts)
                        st.session_state.tfidf_vec = vec
                        st.session_state.tfidf_mat = mat
                        st.session_state.backend = "tfidf"
                        st.success("TF-IDF index built.")

                st.session_state.chunks = all_texts
                st.session_state.metas = all_metas
                st.success("Index ready. Switch to Chat section below.")
            except Exception as e:
                st.error(f"Ingest pipeline failed: {e}")
                logging.error("Ingest pipeline exception:\n" + traceback.format_exc())
                show_startup_log_in_sidebar()

# --------- Chat ----------
st.header("2) Chat — ask questions about ingested docs")
query = st.text_input("Ask anything about ingested documents")
ask_btn = st.button("Ask")

if ask_btn and query.strip():
    if not st.session_state.chunks:
        st.error("No index available. Ingest documents first.")
    else:
        with st.spinner("Preparing answer..."):
            client = st.session_state.client
            short_q = query
            # embed query if possible
            q_emb = None
            try:
                if client and st.session_state.backend == "gemini" and st.session_state.embeddings is not None:
                    resp = client.models.embed_content(model=embed_model_input, contents=[short_q])
                    # robust extraction
                    np = import_numpy()
                    try:
                        q_emb = np.array([e.embedding for e in resp.embeddings], dtype="float32")
                    except Exception:
                        if isinstance(resp, dict) and "embeddings" in resp:
                            q_emb = np.array([e["embedding"] for e in resp["embeddings"]], dtype="float32")
                        else:
                            q_emb = None
                    if q_emb is not None:
                        q_emb = (q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)).astype("float32")
            except Exception:
                logging.error("Query embed failed:\n" + traceback.format_exc())
                q_emb = None

            candidate_idxs = []
            if q_emb is not None and st.session_state.embeddings is not None:
                try:
                    nn = nearest_neighbors_search(q_emb, st.session_state.embeddings, top_k=top_k)
                    candidate_idxs = [idx for idx, _ in nn]
                except Exception:
                    candidate_idxs = list(range(min(10, len(st.session_state.chunks))))
            else:
                # TF-IDF fallback
                try:
                    tf_res = import_sklearn_tfidf()
                    if tf_res and st.session_state.backend == "tfidf":
                        TfidfVectorizer, cosine_similarity = tf_res
                        qv = st.session_state.tfidf_vec.transform([short_q])
                        sims = cosine_similarity(qv, st.session_state.tfidf_mat).squeeze()
                        idxs = np.argsort(-sims)[:top_k]
                        candidate_idxs = [int(i) for i in idxs]
                    else:
                        candidate_idxs = list(range(min(top_k, len(st.session_state.chunks))))
                except Exception:
                    candidate_idxs = list(range(min(top_k, len(st.session_state.chunks))))

            selected_idxs = candidate_idxs
            # build context with char budget
            context_text, used = build_combined_context(st.session_state.chunks, st.session_state.metas, selected_idxs, max_chars=max_context_chars)

            # generation
            answer = None
            try:
                if client:
                    answer = generate_answer_gemini(client, gen_model_input, query, context_text, max_output_tokens=512)
                else:
                    answer = None
            except Exception as e:
                logging.error("Generation failed:\n" + traceback.format_exc())
                answer = None

            st.markdown("### Answer")
            if answer:
                st.write(answer)
            else:
                # extractive fallback
                if not selected_idxs:
                    st.info("No relevant passages found.")
                else:
                    parts = []
                    for idx in selected_idxs[:3]:
                        s = st.session_state.chunks[idx]
                        src = st.session_state.metas[idx].get("source", "unknown")
                        snippet = s if len(s) < 800 else s[:800].rsplit("\n", 1)[0] + "..."
                        parts.append(f"From [{src}]:\n{snippet}")
                    st.write("\n\n".join(parts))

            # show sources expanders
            if st.checkbox("Show sources (expand)"):
                st.markdown("#### Sources / Evidence (expanded)")
                for idx in selected_idxs:
                    src = st.session_state.metas[idx].get("source", "unknown")
                    with st.expander(f"{src}"):
                        st.write(st.session_state.chunks[idx])

# --------- Save / Load Index ----------
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("Save index to ./doc_index"):
        try:
            os.makedirs("doc_index", exist_ok=True)
            import pickle, numpy as _np
            with open("doc_index/chunks.pkl", "wb") as f:
                pickle.dump(st.session_state.chunks, f)
            with open("doc_index/metas.pkl", "wb") as f:
                pickle.dump(st.session_state.metas, f)
            if st.session_state.embeddings is not None:
                _np.save("doc_index/embeddings.npy", st.session_state.embeddings)
            st.success("Saved index to ./doc_index")
        except Exception as e:
            st.error(f"Save failed: {e}\nSee startup log.")
            logging.error("Save index failed:\n" + traceback.format_exc())
with col2:
    if st.button("Load index from ./doc_index"):
        try:
            import pickle, numpy as _np
            st.session_state.chunks = pickle.load(open("doc_index/chunks.pkl", "rb"))
            st.session_state.metas = pickle.load(open("doc_index/metas.pkl", "rb"))
            if os.path.exists("doc_index/embeddings.npy"):
                st.session_state.embeddings = _np.load("doc_index/embeddings.npy", allow_pickle=False)
                st.session_state.backend = "gemini"
                # try to build faiss if available
                try:
                    if import_faiss():
                        st.session_state.faiss_index = build_faiss_index(st.session_state.embeddings)
                except Exception:
                    st.session_state.faiss_index = None
            else:
                st.session_state.embeddings = None
                st.session_state.backend = "tfidf"
            st.success("Loaded index from ./doc_index")
        except Exception as e:
            st.error(f"Load failed: {e}")
            logging.error("Load index failed:\n" + traceback.format_exc())

st.caption("DocQA — Safe RAG. If the app failed during startup, check the Startup Log in the sidebar for full traceback.")
