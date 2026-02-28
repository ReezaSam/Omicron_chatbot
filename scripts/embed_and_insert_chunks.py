import os
import json
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer

# ==========================
# DB CONFIG
# ==========================
DB_NAME = os.getenv("DB_NAME", "omicron_db")
DB_USER = os.getenv("DB_USER", "postgres")

# Default password set to 2719 (can be overridden by env var DB_PASSWORD)
DB_PASSWORD = os.getenv("DB_PASSWORD", "2719")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# IMPORTANT: Use the doc_id that exists in kb_documents
DOC_ID = int(os.getenv("DOC_ID", "1"))

# ==========================
# MODEL (SBERT) => 384 dims
# ==========================
MODEL_NAME = str(Path(__file__).resolve().parent.parent / "models" / "rusl_embedding_model")

def load_blocks(json_path: Path):
    """
    Supports BOTH:
    1) pretty json blocks separated by blank lines (older format)
    2) normal JSONL (one JSON object per line)  <-- recommended
    """
    text = json_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    # Case 1: Pretty JSON blocks separated by blank lines
    if "\n\n{" in text or (text.startswith("{") and "\n\n" in text):
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        objs = []
        for b in blocks:
            try:
                objs.append(json.loads(b))
            except json.JSONDecodeError:
                continue
        if objs:
            return objs

    # Case 2: Normal JSONL
    objs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return objs


def main():
    root = Path(__file__).resolve().parent.parent
    json_path = root / "chunks" / "faculty_handbook_chunks_metadata.jsonl"

    print("📌 Project root:", root)
    print("📌 Looking for chunks file at:", json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Chunks file NOT found: {json_path}")

    # Load chunks
    objs = load_blocks(json_path)
    print(f"✅ Total chunks loaded from file: {len(objs)}")

    if not objs:
        print("⚠️ No chunks found. Exiting.")
        return

    # Load embedding model
    print("📌 Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # Connect DB
    print(f"📌 Connecting to DB {DB_NAME} at {DB_HOST}:{DB_PORT} as {DB_USER} ...")
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    cur = conn.cursor()

    # ==========================
    # CLEAR EXISTING CHUNKS FIRST
    # ==========================
    print(f"🗑️  Clearing existing chunks for doc_id={DOC_ID}...")
    cur.execute("DELETE FROM kb_chunks WHERE doc_id = %s;", (DOC_ID,))
    conn.commit()
    print("✅ Old chunks cleared.")

    # Reset sequence (optional; safe to ignore failure)
    try:
        cur.execute("ALTER SEQUENCE kb_chunks_chunk_id_seq RESTART WITH 1;")
        conn.commit()
        print("✅ ID sequence reset.")
    except Exception as e:
        conn.rollback()
        print("⚠️ Could not reset sequence (safe to ignore if sequence name differs):", e)

    # ==========================
    # PREP TEXTS + METADATA
    # ==========================
    texts = []
    metas = []

    for data in objs:
        text = (data.get("text") or "").strip()
        meta = data.get("metadata", {}) or {}
        section = (meta.get("section") or "UNKNOWN").strip()
        chunk_type = (meta.get("type") or "text").strip().lower()

        if chunk_type not in ("text", "table"):
            chunk_type = "text"

        if not text:
            continue

        texts.append(text)
        metas.append((section, chunk_type))

    print(f"📌 Embedding {len(texts)} chunks in batches...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # ==========================
    # INSERT
    # ==========================
    inserted = 0
    for (section, chunk_type), text, emb in zip(metas, texts, embeddings):
        embedding = emb.tolist()

        cur.execute(
            """
            INSERT INTO kb_chunks (doc_id, section, chunk_type, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (DOC_ID, section, chunk_type, text, embedding),
        )

        inserted += 1
        if inserted % 50 == 0:
            print(f"✅ Inserted {inserted} chunks...")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n🎉 Done! Total chunks inserted: {inserted}")
    print("✅ No duplicates — old chunks were cleared before inserting.")


if __name__ == "__main__":
    main()