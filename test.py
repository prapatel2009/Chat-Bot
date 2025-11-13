# app_gemini_rag.py
# RAG chat using Google Gemini (Google AI Studio API key)
# Run: streamlit run app_gemini_rag.py

import streamlit as st
import io, os, math, pickle
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# text extraction
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# google genai client (Gemini)
try:
    # preferred import style per new SDK
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False

# optional: token-aware chunking
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

# faiss (fast similarity)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

st.set_page_config(page_title="Gemini RAG Chat", layout="wide")
st.title("📘 Gemini RAG Chat — (Google AI Studio)")

# -----------------------
# Helpers: text extraction
# -----------------------
def extract_text_from_pdf_bytes(b: bytes) -> str:
    if not pdfplumber:
        return ""
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                txt = p.extract_text() or ""
                if txt.strip():
                    parts.append(txt.strip())
    except Exception:
        return ""
    return "\n\n".join(parts)

def extract_text_from_docx_bytes(b: bytes) -> str:
    if not docx:
        return ""
    try:
        d = docx.Document(io.BytesIO(b))
        paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paras)
    except Exception:
        return ""

def extract_text_from_excel_bytes(b: bytes) -> str:
    if pd is None:
        return ""
    try:
        df = pd.read_excel(io.BytesIO(b), engine="openpyxl")
        rows = []
        for _, r in df.iterrows():
            rows.append(" | ".join([str(x) for x in r.tolist() if pd.notna(x)]))
        return "\n".join(rows)
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(b))
            rows = []
            for _, r in df.iterrows():
                rows.append(" | ".join([str(x) for x in r.tolist() if pd.notna(x)]))
            return "\n".join(rows)
        except Exception:
            return ""

def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return str(b)

def fetch_webpage_text(url: str) -> str:
    if BeautifulSoup is None:
        return ""
    try:
        import requests
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        if soup.body:
            return soup.body.get_text(separator="\n")
        return soup.get_text(separator="\n")
    except Exception:
        return ""

def clean_text(s: str) -> str:
    return "\n".join([ln.strip() for ln in s.splitlines() if ln.strip()])

# -----------------------
# chunking (token-aware fallback)
# -----------------------
def chunk_text_char_based(text: str, size: int = 1500, overlap: int = 300) -> List[str]:
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
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i:i+size].strip())
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks

def chunk_text_tokenwise(text: str, max_tokens: int = 400, overlap_tokens: int = 80) -> List[str]:
    if not text:
        return []
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            toks = enc.encode(text)
            out = []
            i = 0
            L = len(toks)
            while i < L:
                j = min(i + max_tokens, L)
                out.append(enc.decode(toks[i:j]))
                i = j - overlap_tokens
                if i < 0:
                    i = 0
            return out
        except Exception:
            pass
    # fallback char-based
    char_size = max(400, max_tokens * 4)
    char_overlap = max(50, overlap_tokens * 4)
    return chunk_text_char_based(text, size=char_size, overlap=char_overlap)

# -----------------------
# Google GenAI (Gemini) wrappers
# -----------------------
def init_genai_client(api_key: str):
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK not installed. pip install google-genai")
    # Use Client explicit auth (recommended)
    client = genai.Client(api_key=api_key)
    return client

def embed_texts_gemini(client, texts: List[str], model: str = "gemini-embedding-001") -> np.ndarray:
    # uses client.models.embed_content per docs (batch supported)
    resp = client.models.embed_content(model=model, contents=texts)
    # resp.embeddings is list of vectors
    embs = np.array([e.embedding for e in resp.embeddings], dtype="float32")
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return (embs / norms).astype("float32")

def generate_answer_gemini(client, model: str, prompt: str, max_output_tokens: int = 512) -> str:
    # Use generate_content - pass simple text prompt (we will provide context)
    # The method returns an object with .text property in examples.
    resp = client.models.generate_content(model=model, contents=prompt)
    # many SDK versions provide .text or .candidates[0].content
    # fallback robust extraction:
    try:
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception:
        return str(resp)

# -----------------------
# Vector helpers (FAISS fallback)
# -----------------------
def build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embs)
    index.add(embs)
    return index

def knn_search_faiss(index, emb: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
    D, I = index.search(emb.astype("float32"), top_k)
    # I shape (1, k) D shape (1, k)
    return [(int(I[0,i]), float(D[0,i])) for i in range(I.shape[1])]

# -----------------------
# Session state
# -----------------------
if "chunks" not in st.session_state: st.session_state.chunks = []
if "metas" not in st.session_state: st.session_state.metas = []
if "embeddings" not in st.session_state: st.session_state.embeddings = None
if "faiss_index" not in st.session_state: st.session_state.faiss_index = None
if "tfidf_vec" not in st.session_state: st.session_state.tfidf_vec = None
if "tfidf_mat" not in st.session_state: st.session_state.tfidf_mat = None
if "client" not in st.session_state: st.session_state.client = None

# -----------------------
# Sidebar / settings
# -----------------------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Google AI Studio API key", type="password")
embed_model = st.sidebar.text_input("Embedding model", value="gemini-embedding-001")
gen_model = st.sidebar.text_input("Generation model", value="gemini-2.5-flash-lite-preview-06-17")
chunk_tokens = st.sidebar.number_input("Chunk token size (approx)", min_value=128, max_value=2000, value=600)
overlap_tokens = st.sidebar.number_input("Overlap tokens", min_value=0, max_value=400, value=80)
top_k = st.sidebar.slider("Top K retrieved chunks", min_value=3, max_value=30, value=10)
max_context_tokens = st.sidebar.number_input("Max context tokens to send", min_value=256, max_value=4000, value=1500)
st.sidebar.markdown("---")
st.sidebar.markdown("Notes: set your Google API key above and click 'Init client'.")

if st.sidebar.button("Init client"):
    if not api_key:
        st.sidebar.error("Provide API key first.")
    else:
        try:
            client = init_genai_client(api_key.strip())
            st.session_state.client = client
            st.sidebar.success("GenAI client initialized.")
        except Exception as e:
            st.sidebar.error(f"Init failed: {e}")

# -----------------------
# Upload & Ingest
# -----------------------
st.header("1) Upload documents / URL")
uploaded = st.file_uploader("Upload PDF/DOCX/XLSX/CSV/TXT (multiple)", accept_multiple_files=True, type=["pdf","docx","xlsx","csv","txt"])
url = st.text_input("Optional web/Confluence URL")
if st.button("Ingest & build"):
    all_texts = []
    all_metas = []
    for f in uploaded or []:
        name = f.name
        b = f.read()
        ext = name.lower().split(".")[-1]
        st.info(f"Extracting {name} ...")
        txt = ""
        if ext == "pdf":
            txt = extract_text_from_pdf_bytes(b)
        elif ext == "docx":
            txt = extract_text_from_docx_bytes(b)
        elif ext in ("xlsx","xls"):
            txt = extract_text_from_excel_bytes(b)
        elif ext in ("csv","txt"):
            txt = extract_text_from_txt_bytes(b)
        else:
            txt = extract_text_from_txt_bytes(b)
        txt = clean_text(txt)
        if not txt:
            st.warning(f"No text found in {name}")
            continue
        chunks = chunk_text_tokenwise(txt, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        for i,c in enumerate(chunks):
            all_texts.append(c)
            all_metas.append({"source": name, "chunk_id": i, "ingested_at": datetime.utcnow().isoformat()})
    if url:
        st.info("Fetching URL ...")
        page = fetch_webpage_text(url)
        page = clean_text(page)
        if page:
            chunks = chunk_text_tokenwise(page, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            for i,c in enumerate(chunks):
                all_texts.append(c)
                all_metas.append({"source": url, "chunk_id": i, "ingested_at": datetime.utcnow().isoformat()})

    if not all_texts:
        st.error("No text ingested.")
    else:
        st.success(f"Ingested {len(all_texts)} chunks.")
        # embeddings via Gemini if client initialized
        if st.session_state.client is not None:
            try:
                st.info("Creating embeddings (Gemini)...")
                embs = embed_texts_gemini(st.session_state.client, all_texts, model=embed_model)
                st.session_state.embeddings = embs
                st.session_state.metas = all_metas
                st.session_state.chunks = all_texts
                st.success("Embeddings created.")
                if FAISS_AVAILABLE:
                    try:
                        idx = build_faiss_index(embs)
                        st.session_state.faiss_index = idx
                        st.success("FAISS index built.")
                    except Exception as e:
                        st.warning(f"FAISS build failed: {e}")
                else:
                    st.info("FAISS not available — will use numpy/TF-IDF fallback.")
            except Exception as e:
                st.error(f"Embeddings failed: {e}")
                # fallback to TF-IDF
                vec = TfidfVectorizer(max_features=65536, stop_words="english")
                mat = vec.fit_transform(all_texts)
                st.session_state.tfidf_vec = vec
                st.session_state.tfidf_mat = mat
                st.session_state.backend = "tfidf"
                st.session_state.chunks = all_texts
                st.session_state.metas = all_metas
                st.success("Built TF-IDF fallback index.")
        else:
            # client not initialized -> TF-IDF fallback
            vec = TfidfVectorizer(max_features=65536, stop_words="english")
            mat = vec.fit_transform(all_texts)
            st.session_state.tfidf_vec = vec
            st.session_state.tfidf_mat = mat
            st.session_state.backend = "tfidf"
            st.session_state.chunks = all_texts
            st.session_state.metas = all_metas
            st.success("Built TF-IDF fallback index (no API key).")

# -----------------------
# Chat UI
# -----------------------
st.header("2) Chat (ask questions)")
q = st.text_input("Ask a question about the uploaded docs")
ask = st.button("Ask")
if ask and q.strip():
    if not st.session_state.chunks:
        st.error("Index empty — ingest docs first.")
    else:
        with st.spinner("Searching and generating answer..."):
            short_q = q
            # embed question (Gemini) if possible
            q_emb = None
            if st.session_state.client is not None and st.session_state.embeddings is not None:
                try:
                    q_embs = st.session_state.client.models.embed_content(model=embed_model, contents=[short_q])
                    q_emb = np.array([e.embedding for e in q_embs.embeddings], dtype="float32")
                    # normalize
                    q_emb = (q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)).astype("float32")
                except Exception:
                    q_emb = None

            candidate_idxs = []
            if q_emb is not None and st.session_state.embeddings is not None and FAISS_AVAILABLE and st.session_state.faiss_index is not None:
                # FAISS search
                hits = knn_search_faiss(st.session_state.faiss_index, q_emb, top_k=top_k)
                candidate_idxs = [i for i,_ in hits]
            elif q_emb is not None and st.session_state.embeddings is not None:
                # brute-force dot product
                sims = (st.session_state.embeddings @ q_emb.T).squeeze()
                idxs = np.argsort(-sims)[:top_k]
                candidate_idxs = [int(i) for i in idxs]
            else:
                # TF-IDF fallback
                if st.session_state.tfidf_vec is not None and st.session_state.tfidf_mat is not None:
                    qv = st.session_state.tfidf_vec.transform([short_q])
                    sims = cosine_similarity(qv, st.session_state.tfidf_mat).squeeze()
                    idxs = np.argsort(-sims)[:top_k]
                    candidate_idxs = [int(i) for i in idxs]
                else:
                    candidate_idxs = list(range(min(top_k, len(st.session_state.chunks))))

            # Build context from top candidates until token budget
            selected = candidate_idxs
            # Simple concatenation (you can improve by token counting/reduction)
            context_parts = []
            used_sources = []
            for idx in selected:
                txt = st.session_state.chunks[idx]
                src = st.session_state.metas[idx].get("source", "unknown")
                context_parts.append(f"[{src}] {txt}")
                used_sources.append(src)
            context_text = "\n\n---\n\n".join(context_parts)
            if len(context_text) > max_context_tokens * 4:
                context_text = context_text[:max_context_tokens * 4]

            # Build prompt instructing Gemini to use only context
            prompt = (
                "You are a helpful assistant. Answer using ONLY the CONTEXT below. If the answer is not present, say 'I don't know'. "
                "Cite the source(s) in square brackets.\n\n"
                f"CONTEXT:\n{context_text}\n\nQUESTION:\n{q}\n\nAnswer concisely and cite sources."
            )

            answer = None
            if st.session_state.client is not None:
                try:
                    resp = st.session_state.client.models.generate_content(model=gen_model, contents=prompt)
                    # robustly extract text
                    answer = resp.text if hasattr(resp, "text") else str(resp)
                except Exception as e:
                    st.warning(f"Generation failed: {e}")
                    answer = None

            # If no API / generation failed -> extractive fallback
            st.markdown("### Answer")
            if answer:
                st.write(answer)
            else:
                # show extractive snippets
                snippets = []
                for idx in selected[:3]:
                    s = st.session_state.chunks[idx]
                    src = st.session_state.metas[idx].get("source","unknown")
                    snippet = s if len(s) < 800 else s[:800].rsplit("\n",1)[0] + "..."
                    snippets.append(f"From [{src}]:\n{snippet}")
                st.write("\n\n".join(snippets))

            # show sources toggle
            if st.checkbox("Show sources / evidence"):
                for idx in selected:
                    src = st.session_state.metas[idx].get("source","unknown")
                    with st.expander(f"{src}"):
                        st.write(st.session_state.chunks[idx])

# -----------------------
# Save / load index
# -----------------------
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("Save index to ./gemini_index"):
        os.makedirs("gemini_index", exist_ok=True)
        pickle.dump(st.session_state.chunks, open("gemini_index/chunks.pkl","wb"))
        pickle.dump(st.session_state.metas, open("gemini_index/metas.pkl","wb"))
        if st.session_state.embeddings is not None:
            np.save("gemini_index/embeddings.npy", st.session_state.embeddings)
        st.success("Saved index.")
with col2:
    if st.button("Load index from ./gemini_index"):
        try:
            st.session_state.chunks = pickle.load(open("gemini_index/chunks.pkl","rb"))
            st.session_state.metas = pickle.load(open("gemini_index/metas.pkl","rb"))
            if os.path.exists("gemini_index/embeddings.npy"):
                st.session_state.embeddings = np.load("gemini_index/embeddings.npy", allow_pickle=False)
                if FAISS_AVAILABLE:
                    st.session_state.faiss_index = build_faiss_index(st.session_state.embeddings)
            st.success("Loaded index.")
        except Exception as e:
            st.error(f"Load failed: {e}")

st.caption("Built with Google Gemini embeddings & generation (use your Google AI Studio API key). See Google GenAI docs for details.")
