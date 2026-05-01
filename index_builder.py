#!/usr/bin/env python3
"""
index_builder.py - Run this ONCE before starting chatbot_api.py
Builds FAISS indexes for all universities with progress display.

Usage:
    python3 index_builder.py                          # all universities
    python3 index_builder.py demiroglu_bilim_university   # one university
"""

import os, sys, json, time, numpy as np, faiss, openai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

BASE       = Path(__file__).parent
DATA_DIR   = BASE / "data"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
BATCH_SIZE    = 100   # bigger batches = faster


def chunk_text(text):
    text = text.replace("\\n", "\n").strip()
    if not text:
        return []
    result, start = [], 0
    while start < len(text):
        c = text[start:start + CHUNK_SIZE].strip()
        if len(c) > 40:
            result.append(c)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return result


def embed_batch(texts):
    clean = [t.replace("\n", " ")[:8000] for t in texts]
    for attempt in range(3):
        try:
            resp = openai.embeddings.create(input=clean, model=EMBED_MODEL)
            return [item.embedding for item in resp.data]
        except openai.RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"  Rate limit - waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Rate limit: 3 retries exhausted")


def build(uni_id):
    uni_dir   = DATA_DIR / uni_id
    cache_dir = OUTPUT_DIR / uni_id

    if not uni_dir.exists():
        print(f"[ERROR] Not found: {uni_dir}")
        return False

    # Already built?
    if (cache_dir / "chunks.json").exists() and \
       (cache_dir / "faiss_index.bin").exists() and \
       (cache_dir / "embeddings.npy").exists():
        print(f"[SKIP] {uni_id} - already indexed (delete output/{uni_id}/ to rebuild)")
        return True

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  {uni_id}")
    print(f"{'='*55}")

    # Load pages
    all_files = sorted(uni_dir.glob("**/*.json"))
    print(f"  Loading {len(all_files)} JSON files...")
    pages = []
    for p in all_files:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("content") and data.get("title"):
                pages.append(data)
        except Exception:
            pass
    print(f"  Valid pages: {len(pages)}")

    if not pages:
        print("  [ERROR] No valid pages!")
        return False

    display_name = pages[0].get("university", uni_id.replace("_", " ").title())

    # Chunk
    chunks = []
    for page in pages:
        for text in chunk_text(page["content"]):
            chunks.append({
                "content": text,
                "metadata": {
                    "title":        page.get("title", ""),
                    "url":          page.get("url", ""),
                    "language":     page.get("language", ""),
                    "content_type": page.get("content_type", ""),
                    "university":   display_name,
                }
            })
    print(f"  Chunks: {len(chunks)}")

    # Embed
    texts     = [c["content"] for c in chunks]
    total     = len(texts)
    all_vecs  = []
    t_start   = time.time()

    print(f"  Embedding {total} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        batch   = texts[i:i + BATCH_SIZE]
        vecs    = embed_batch(batch)
        all_vecs.extend(vecs)

        done    = min(i + BATCH_SIZE, total)
        pct     = done / total * 100
        elapsed = time.time() - t_start
        eta     = (elapsed / done * (total - done)) if done > 0 else 0

        bar_len = 30
        filled  = int(bar_len * done / total)
        bar     = "#" * filled + "-" * (bar_len - filled)
        print(f"  [{bar}] {done}/{total} ({pct:.0f}%)  ETA: {eta:.0f}s", end="\r")
        time.sleep(0.1)

    print(f"\n  Embedding done in {time.time()-t_start:.0f}s")

    # Build FAISS
    emb_np = np.array(all_vecs, dtype="float32")
    dim    = emb_np.shape[1]
    index  = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb_np)
    index.add(emb_np)
    print(f"  FAISS: {index.ntotal} vectors, dim={dim}")

    # Save
    with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    faiss.write_index(index, str(cache_dir / "faiss_index.bin"))
    np.save(str(cache_dir / "embeddings.npy"), emb_np)

    size_mb = sum(
        (cache_dir / fn).stat().st_size
        for fn in ["chunks.json", "faiss_index.bin", "embeddings.npy"]
    ) / 1024 / 1024
    print(f"  Saved to output/{uni_id}/  ({size_mb:.1f} MB)")
    return True


def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        if not DATA_DIR.exists():
            print(f"ERROR: data/ not found at {DATA_DIR}")
            sys.exit(1)
        targets = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())

    print(f"Universities to index: {targets}")
    print(f"Output dir: {OUTPUT_DIR}\n")

    t0 = time.time()
    ok, fail = [], []
    for uid in targets:
        if build(uid):
            ok.append(uid)
        else:
            fail.append(uid)

    total_time = time.time() - t0
    print(f"\n{'='*55}")
    print(f"DONE in {total_time/60:.1f} min")
    print(f"  OK   ({len(ok)}): {ok}")
    if fail:
        print(f"  FAIL ({len(fail)}): {fail}")
    print(f"\nNow start the server:")
    print(f"  python3 chatbot_api.py")


if __name__ == "__main__":
    main()