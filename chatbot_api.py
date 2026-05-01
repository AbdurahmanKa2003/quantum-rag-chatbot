#!/usr/bin/env python3
"""
chatbot_api.py — Multi-University RAG Chatbot v4.0
===================================================
NO separate build step needed.

Two data source types are supported automatically:

  TYPE A — IZU-style pre-chunked file (your existing chunks.json):
    {
      "chunk_id": ..., "content": ...,
      "metadata": {"url": ..., "title": ..., "language": ...}
    }

  TYPE B — Raw scraped JSON files in data/<university>/{en,tr}/*.json:
    {
      "title": ..., "url": ..., "content": ...,
      "language": ..., "university": ...
    }
    → chunked automatically at startup, FAISS index built in memory.
    → optionally cached to disk so next startup is instant.

Usage:
    python chatbot_api.py
"""

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
import re
import numpy as np
import faiss
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import logging
import time

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

BASE = Path(__file__).parent   # project root

# ---
#  UNIVERSITY CONFIG
#  Add/remove universities here. The loader figures out the format automatically.
# ---
UNIVERSITIES: List[Dict] = [

    # --- TYPE A: IZU - pre-chunked chunks.json + pre-built faiss_index.bin ---
    {
        "id":    "izu",
        "name":  "İstanbul Sabahattin Zaim Üniversitesi",
        "short": "İZÜ",
        "color": "#8b2232",
        # Pre-built files (already exist in your project root):
        "chunks_path":     str(BASE / "chunks.json"),
        "index_path":      str(BASE / "faiss_index.bin"),
        "embeddings_path": str(BASE / "embeddings_openai_izu.npy"),
        # No data_dir needed - using pre-built files above
    },

    # --- TYPE B: Raw scraped folder - built automatically at startup ---
    # Uncomment and fill in for each new university.
    # The loader will chunk + embed + build FAISS index on first run,
    # then cache to disk so next startup is instant.

    {
        "id":       "demiroglu_bilim_university",
        "name":     "Demiro-lu Bilim -niversitesi",
        "short":    "DB-",
        "color":    "#1565c0",
        "data_dir": str(BASE / "data" / "demiroglu_bilim_university"),
        # cache_dir is optional - defaults to BASE/output/<id>/
    },
    {
        "id":       "dogus_university",
        "name":     "Do-u- -niversitesi",
        "short":    "D-",
        "color":    "#2e7d32",
        "data_dir": str(BASE / "data" / "dogus_university"),
    },
    {
        "id":       "fenerbahce_university",
        "name":     "Fenerbah-e -niversitesi",
        "short":    "FB-",
        "color":    "#1a5276",
        "data_dir": str(BASE / "data" / "fenerbahce_university"),
    },
    {
        "id":       "halic_university",
        "name":     "Hali- -niversitesi",
        "short":    "H-",
        "color":    "#6a1b9a",
        "data_dir": str(BASE / "data" / "halic_university"),
    },
    {
        "id":       "ibn_haldun_university",
        "name":     "-bn Haldun -niversitesi",
        "short":    "-H-",
        "color":    "#e65100",
        "data_dir": str(BASE / "data" / "ibn_haldun_university"),
    },
    {
        "id":       "istanbul_29_mayis_university",
        "name":     "-stanbul 29 May-s -niversitesi",
        "short":    "29M-",
        "color":    "#00695c",
        "data_dir": str(BASE / "data" / "istanbul_29_mayis_university"),
    },
    {
        "id":       "istanbul_beykent_university",
        "name":     "-stanbul Beykent -niversitesi",
        "short":    "B-",
        "color":    "#ad1457",
        "data_dir": str(BASE / "data" / "istanbul_beykent_university"),
    },
    {
        "id":       "istanbul_esenyurt_university",
        "name":     "-stanbul Esenyurt -niversitesi",
        "short":    "-E-",
        "color":    "#4e342e",
        "data_dir": str(BASE / "data" / "istanbul_esenyurt_university"),
    },
    {
        "id":       "istanbul_galata_university",
        "name":     "-stanbul Galata -niversitesi",
        "short":    "GAL",
        "color":    "#37474f",
        "data_dir": str(BASE / "data" / "istanbul_galata_university"),
    },
    {
        "id":       "istanbul_health_and_technology_university",
        "name":     "-stanbul Sa-l-k ve Teknoloji -niversitesi",
        "short":    "ISTU",
        "color":    "#558b2f",
        "data_dir": str(BASE / "data" / "istanbul_health_and_technology_university"),
    },
    {
        "id":       "istanbul_kent_university",
        "name":     "-stanbul Kent -niversitesi",
        "short":    "KENT",
        "color":    "#f57f17",
        "data_dir": str(BASE / "data" / "istanbul_kent_university"),
    },
    {
        "id":       "istanbul_okan_university",
        "name":     "-stanbul Okan -niversitesi",
        "short":    "OKAN",
        "color":    "#283593",
        "data_dir": str(BASE / "data" / "istanbul_okan_university"),
    },
    {
        "id":       "istanbul_yeni_yuzyil_university",
        "name":     "-stanbul Yeni Y-zy-l -niversitesi",
        "short":    "YY-",
        "color":    "#880e4f",
        "data_dir": str(BASE / "data" / "istanbul_yeni_yuzyil_university"),
    },
    {
        "id":       "kadir_has_university",
        "name":     "Kadir Has -niversitesi",
        "short":    "KH-",
        "color":    "#00838f",
        "data_dir": str(BASE / "data" / "kadir_has_university"),
    },
    {
        "id":       "piri_reis_university",
        "name":     "Piri Reis -niversitesi",
        "short":    "PR-",
        "color":    "#4527a0",
        "data_dir": str(BASE / "data" / "piri_reis_university"),
    },
    {
        "id":       "turkish_japanese_science_and_technology_university",
        "name":     "T-rk-Japon Bilim ve Teknoloji -niversitesi",
        "short":    "TJSTU",
        "color":    "#c62828",
        "data_dir": str(BASE / "data" / "turkish_japanese_science_and_technology_university"),
    },
    {
        "id":       "uskudar_university",
        "name":     "-sk-dar -niversitesi",
        "short":    "-SK",
        "color":    "#0277bd",
        "data_dir": str(BASE / "data" / "uskudar_university"),
    },
]

# --- Tuneable constants ---
EMBED_MODEL      = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o-mini"
TOP_K            = 8
SCORE_THRESHOLD  = 0.25      # cosine similarity minimum
MAX_CHARS        = 12_000    # context window cap
EMBED_CACHE_SIZE = 512
CHUNK_SIZE       = 800       # chars for TYPE B chunking
CHUNK_OVERLAP    = 150
EMBED_BATCH      = 50        # pages per OpenAI batch for TYPE B

# --- Global RAG store ---
rag_store: Dict[str, Dict] = {}

# ---
#  LOADERS
# ---

def _stamp_chunks(chunks: list, uni: Dict) -> list:
    """Add university metadata fields to every chunk."""
    for c in chunks:
        c.setdefault("metadata", {})
        c["metadata"].update({
            "university_id":    uni["id"],
            "university_name":  uni["name"],
            "university_short": uni["short"],
            "university_color": uni["color"],
        })
    return chunks


# --- TYPE A: pre-built files ---
def _load_prebuilt(uni: Dict) -> bool:
    """Load IZU-style pre-built chunks.json + faiss_index.bin."""
    uid = uni["id"]
    try:
        for key in ("chunks_path", "index_path", "embeddings_path"):
            if not Path(uni[key]).exists():
                raise FileNotFoundError(f"{key} missing: {uni[key]}")

        with open(uni["chunks_path"], encoding="utf-8") as f:
            chunks = json.load(f)

        # Normalise: IZU chunks already have metadata, just stamp university fields
        _stamp_chunks(chunks, uni)

        index = faiss.read_index(uni["index_path"])
        embeddings = np.load(uni["embeddings_path"])

        rag_store[uid] = {"chunks": chunks, "faiss_index": index,
                          "embeddings": embeddings, "meta": uni}
        logger.info("✓ [TYPE A] %s: %d chunks, %d vectors", uni["short"],
                    len(chunks), index.ntotal)
        return True

    except Exception as e:
        logger.error("✗ [TYPE A] %s failed: %s", uid, e)
        return False


# --- TYPE B: raw scraped folder ---
def _chunk_text(text: str) -> List[str]:
    text = text.replace("\\n", "\n").strip()
    if not text:
        return []
    result, start = [], 0
    while start < len(text):
        result.append(text[start:start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in result if len(c) > 40]


def _embed_batch_raw(texts: List[str]) -> List[List[float]]:
    clean = [t.replace("\n", " ")[:8000] for t in texts]
    for attempt in range(3):
        try:
            resp = openai.embeddings.create(input=clean, model=EMBED_MODEL)
            return [item.embedding for item in resp.data]
        except openai.RateLimitError:
            wait = 20 * (attempt + 1)
            logger.warning("Rate limited — waiting %ds…", wait)
            time.sleep(wait)
    raise RuntimeError("OpenAI rate limit: 3 retries exhausted")


def _load_raw_folder(uni: Dict) -> bool:
    """
    Load from raw scraped JSON folder.
    Uses disk cache (output/<id>/) so first run embeds,
    subsequent runs load instantly from cache.
    """
    uid      = uni["id"]
    data_dir = Path(uni["data_dir"])
    cache_dir = Path(uni.get("cache_dir", str(BASE / "output" / uid)))

    cache_chunks = cache_dir / "chunks.json"
    cache_index  = cache_dir / "faiss_index.bin"
    cache_emb    = cache_dir / "embeddings.npy"

    # --- Use cache if available ---
    if cache_chunks.exists() and cache_index.exists() and cache_emb.exists():
        try:
            with open(cache_chunks, encoding="utf-8") as f:
                chunks = json.load(f)
            _stamp_chunks(chunks, uni)
            index = faiss.read_index(str(cache_index))
            embeddings = np.load(str(cache_emb))
            rag_store[uid] = {"chunks": chunks, "faiss_index": index,
                              "embeddings": embeddings, "meta": uni}
            logger.info("✓ [TYPE B cache] %s: %d chunks", uni["short"], len(chunks))
            return True
        except Exception as e:
            logger.warning("Cache load failed for %s (%s) — rebuilding…", uid, e)

    # --- Build from raw files ---
    if not data_dir.exists():
        logger.error("✗ data_dir not found: %s", data_dir)
        return False

    logger.info("Building index for %s from %s …", uni["short"], data_dir)

    # Load pages
    pages = []
    for json_path in sorted(data_dir.glob("**/*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                page = json.load(f)
            if page.get("content") and page.get("title"):
                pages.append(page)
        except Exception as e:
            logger.debug("Skip %s: %s", json_path.name, e)

    if not pages:
        logger.error("✗ No valid pages in %s", data_dir)
        return False

    logger.info("  %d pages → chunking…", len(pages))

    # Chunk
    chunks = []
    display_name = pages[0].get("university", uni["name"])
    for page in pages:
        for text in _chunk_text(page["content"]):
            chunks.append({
                "content": text,
                "metadata": {
                    "title":        page.get("title", ""),
                    "url":          page.get("url", ""),
                    "language":     page.get("language", ""),
                    "content_type": page.get("content_type", ""),
                    "university":   display_name,
                },
            })

    logger.info("  %d chunks → embedding…", len(chunks))

    # Embed in batches
    texts = [c["content"] for c in chunks]
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        all_vecs.extend(_embed_batch_raw(batch))
        logger.info("    %d/%d embedded", min(i + EMBED_BATCH, len(texts)), len(texts))
        time.sleep(0.3)

    embeddings_np = np.array(all_vecs, dtype="float32")

    # Build FAISS
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    # Save cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_chunks, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        faiss.write_index(index, str(cache_index))
        np.save(str(cache_emb), embeddings_np)
        logger.info("  ✓ Cached to %s", cache_dir)
    except Exception as e:
        logger.warning("  Cache save failed: %s", e)

    _stamp_chunks(chunks, uni)
    rag_store[uid] = {"chunks": chunks, "faiss_index": index,
                      "embeddings": embeddings_np, "meta": uni}
    logger.info("✓ [TYPE B] %s: %d chunks, %d vectors", uni["short"],
                len(chunks), index.ntotal)
    return True


def _load_university(uni: Dict) -> bool:
    """Detect type and load accordingly."""
    if "data_dir" in uni:
        return _load_raw_folder(uni)
    else:
        return _load_prebuilt(uni)


# ---
#  FASTAPI APP
# ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    for uni in UNIVERSITIES:
        _load_university(uni)
    if not rag_store:
        raise RuntimeError("No universities loaded — check paths in UNIVERSITIES config.")
    logger.info("✓ Ready. Loaded: %s", list(rag_store.keys()))
    yield
    rag_store.clear()


app = FastAPI(
    title="Multi-University RAG Chatbot API",
    description="AI chatbot for multiple Turkish universities",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Pydantic models ---
class ChatRequest(BaseModel):
    message:         str            = Field(..., min_length=1, max_length=500)
    language:        Optional[str]  = "auto"   # "tr" | "en" | "auto"
    university_id:   Optional[str]  = None     # None → all universities
    conversation_id: Optional[str]  = None

class SearchRequest(BaseModel):
    query:         str           = Field(..., min_length=1, max_length=500)
    university_id: Optional[str] = None
    top_k:         int           = Field(default=6, ge=1, le=20)

class ChatResponse(BaseModel):
    answer:            str
    sources:           List[dict]
    response_time_ms:  float
    conversation_id:   str
    university_filter: Optional[str]

class UniversityInfo(BaseModel):
    id: str; name: str; short: str; color: str
    loaded: bool; chunk_count: int

class HealthResponse(BaseModel):
    status: str; universities: List[UniversityInfo]; timestamp: str

# --- Embedding (LRU cached) ---
@lru_cache(maxsize=EMBED_CACHE_SIZE)
def _get_embedding_cached(text: str) -> tuple:
    resp = openai.embeddings.create(
        input=[text.replace("\n", " ")], model=EMBED_MODEL)
    return tuple(resp.data[0].embedding)

def get_embedding(text: str) -> np.ndarray:
    return np.array(_get_embedding_cached(text), dtype="float32")

# --- Retrieval ---
def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    university_id: Optional[str] = None,
) -> List[dict]:
    query_vec = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    targets = (
        {university_id: rag_store[university_id]}
        if university_id and university_id in rag_store
        else rag_store
    )

    candidates = []
    for uid, store in targets.items():
        idx   = store["faiss_index"]
        chnks = store["chunks"]
        k     = min(top_k * 2, idx.ntotal)
        scores, indices = idx.search(query_vec, k)
        for i, s in zip(indices[0], scores[0]):
            if 0 <= i < len(chnks) and s >= SCORE_THRESHOLD:
                candidates.append({
                    "content":  chnks[i]["content"],
                    "metadata": chnks[i]["metadata"],
                    "score":    float(s),
                })

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Diversity deduplication
    seen, diverse = set(), []
    for c in candidates:
        prefix = c["content"][:120].strip()
        if prefix not in seen:
            seen.add(prefix)
            diverse.append(c)
        if len(diverse) >= top_k:
            break
    return diverse

# --- Language detection ---
_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TR_WORDS = {"nedir","nasıl","ne","hangi","kaç","kim","nerede","nereye",
             "ücret","fakülte","bölüm","program","başvuru","için","olan",
             "var","mı","mi","mu","mü","ile","kadar","gibi","veya","neden"}
_EN_WORDS = {"what","how","where","when","which","who","why","faculty",
             "department","program","tuition","fee","admission","application",
             "the","is","are","can","do","does"}

def detect_language(text: str) -> str:
    if any(c in text for c in _TR_CHARS):
        return "tr"
    lower = text.lower()
    tr = sum(1 for w in _TR_WORDS if w in lower)
    en = sum(1 for w in _EN_WORDS if w in lower)
    return "tr" if tr >= en else "en"

# --- Answer generation ---
def _build_context(chunks: List[dict]) -> str:
    parts, total = [], 0
    for i, c in enumerate(chunks, 1):
        uni   = c["metadata"].get("university_short", "?")
        title = c["metadata"].get("title", "Kaynak")
        block = f"[{i}] [{uni}] {title}\n{c['content']}"
        if total + len(block) > MAX_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)

def _clean_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__",    r"\1", text)
    text = re.sub(r"\*(.+?)\*",    r"\1", text)
    text = re.sub(r"_(.+?)_",      r"\1", text)
    text = re.sub(r"^[\-\*•]\s+",  "", text, flags=re.MULTILINE)
    text = re.sub(r"```.*?```",    "", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`",      r"\1", text)
    text = re.sub(r"\n{3,}",       "\n\n", text)
    return text.strip()

def generate_answer(
    query: str,
    chunks: List[dict],
    language: str = "auto",
    university_id: Optional[str] = None,
) -> str:
    if not chunks:
        return (
            "Bu konuda bilgi bulunamadı. Lütfen soruyu farklı şekilde sormayı deneyin."
            if (language == "auto" and detect_language(query) == "tr") or language == "tr"
            else "No relevant information found. Please try rephrasing your question."
        )

    lang    = language if language in ("tr", "en") else detect_language(query)
    context = _build_context(chunks)
    multi   = university_id is None and len(rag_store) > 1

    if lang == "tr":
        system = f"""Sen bir üniversite bilgi asistanısın.
{"Birden fazla üniversitenin verileri mevcut; gerektiğinde hangi bilginin hangi üniversiteye ait olduğunu belirt." if multi else ""}

KURALLAR:
1. YALNIZCA aşağıdaki kaynaklardaki bilgileri kullan. Asla uydurma.
2. Birden fazla kaynakta bilgi varsa sentezle.
3. Liste sorularında TÜM kaynaklardaki maddeleri topla.
4. Evet/Hayır soruları: kaynakta varsa "Evet"/"Hayır" ile başla; yoksa "Bu bilgiye ulaşılamadı" de.
5. Kısa ve öz cevaplar ver (max 5-6 cümle veya madde).
6. Düz metin; Markdown, kalın veya başlık KULLANMA.
7. Madde listeleri için "1. 2. 3." formatını kullan."""

        user = f"Kaynaklar:\n{context}\n\nSoru: {query}"

    else:
        system = f"""You are a university information assistant.
{"Multiple universities' data is available; attribute info to the correct university when relevant." if multi else ""}

RULES:
1. Use ONLY information from sources below. Never fabricate.
2. Synthesise if multiple sources are relevant.
3. For lists, aggregate ALL items across ALL sources.
4. Yes/No: start with "Yes"/"No" if source confirms; else "This information is not available."
5. Concise answers (max 5-6 sentences or bullets).
6. Plain text only — no Markdown, bold, headers.
7. Use "1. 2. 3." for lists."""

        user = f"Sources:\n{context}\n\nQuestion: {query}"

    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.05, max_tokens=350, top_p=0.9,
        frequency_penalty=0.4, presence_penalty=0.1,
    )
    return _clean_markdown(resp.choices[0].message.content)

def _build_sources(chunks: List[dict]) -> List[dict]:
    return [{
        "title":            c["metadata"].get("title", "Unknown"),
        "url":              c["metadata"].get("url", ""),
        "score":            round(c["score"], 4),
        "snippet":          c["content"][:200] + "…",
        "university_id":    c["metadata"].get("university_id", ""),
        "university_short": c["metadata"].get("university_short", ""),
        "university_color": c["metadata"].get("university_color", "#888"),
    } for c in chunks]

# ---
#  ENDPOINTS
# ---
def _uni_info(u: Dict) -> UniversityInfo:
    uid = u["id"]
    return UniversityInfo(
        id=uid, name=u["name"], short=u["short"], color=u["color"],
        loaded=(uid in rag_store),
        chunk_count=len(rag_store[uid]["chunks"]) if uid in rag_store else 0,
    )

@app.get("/")
async def root():
    return {"message": "Multi-University RAG API v4.0",
            "endpoints": {"/chat": "POST", "/search": "POST",
                          "/universities": "GET", "/health": "GET", "/docs": "GET"}}

@app.get("/universities", response_model=List[UniversityInfo])
async def list_universities():
    return [_uni_info(u) for u in UNIVERSITIES]

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if rag_store else "not ready",
        universities=[_uni_info(u) for u in UNIVERSITIES],
        timestamp=datetime.now().isoformat(),
    )

@app.post("/search", response_model=dict)
async def search(req: SearchRequest):
    if req.university_id and req.university_id not in rag_store:
        raise HTTPException(400, f"Unknown university_id: {req.university_id}")
    chunks = retrieve_chunks(req.query, top_k=req.top_k, university_id=req.university_id)
    return {"results": _build_sources(chunks), "count": len(chunks)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.university_id and req.university_id not in rag_store:
        raise HTTPException(400, f"Unknown university_id: {req.university_id}")
    logger.info("Query: %.80s | uni=%s | lang=%s", req.message, req.university_id, req.language)
    t0 = time.perf_counter()
    chunks = retrieve_chunks(req.message, top_k=TOP_K, university_id=req.university_id)
    answer = generate_answer(req.message, chunks,
                             language=req.language or "auto",
                             university_id=req.university_id)
    return ChatResponse(
        answer=answer,
        sources=_build_sources(chunks),
        response_time_ms=(time.perf_counter() - t0) * 1000,
        conversation_id=req.conversation_id or f"conv_{int(time.time())}",
        university_filter=req.university_id,
    )

@app.get("/stats")
async def stats():
    return {
        "total_chunks":        sum(len(s["chunks"]) for s in rag_store.values()),
        "universities_loaded": list(rag_store.keys()),
        "embedding_model":     EMBED_MODEL,
        "llm_model":           CHAT_MODEL,
        "score_threshold":     SCORE_THRESHOLD,
        "embedding_cache":     f"{_get_embedding_cached.cache_info().currsize}/{EMBED_CACHE_SIZE}",
    }

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  MULTI-UNIVERSITY RAG CHATBOT v4.0")
    print("="*60)
    print("  http://localhost:8000")
    print("  docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")