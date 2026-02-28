from pathlib import Path
import re
import pandas as pd
from dotenv import load_dotenv

from Rag_Retriever import get_connection, retrieve_chunks

HERE = Path(__file__).resolve()
BASE = HERE.parent.parent
load_dotenv(BASE / ".env")

QA_CSV = BASE / "data" / "qa_pairs.csv"
OUT = BASE / "scripts" / "eval_rag_outputs_v1"
OUT.mkdir(parents=True, exist_ok=True)

K_EVAL = 5
K_RETRIEVE = 15

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","was","were",
    "be","been","being","it","this","that","these","those","as","at","from","into","about"
}
MONTHS = {"january","february","march","april","may","june","july","august",
          "september","october","november","december"}

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokenize(s: str):
    s = norm(s)
    s = re.sub(r"[^a-z0-9/\s]", " ", s)  # keep slash
    return [t for t in s.split() if t and t not in STOPWORDS]

def keywords_from_answer(ans: str):
    toks = tokenize(ans)
    keys = []
    for t in toks:
        if t.isdigit() or t in MONTHS or "/" in t or len(t) >= 6:
            keys.append(t)
    return list(dict.fromkeys(keys)) if keys else list(dict.fromkeys(toks))

def chunk_text_of(ch):
    if isinstance(ch, tuple) and len(ch) >= 1:
        return ch[0] or ""
    if isinstance(ch, dict):
        return ch.get("chunk_text", "") or ch.get("text", "") or ""
    return ""

def keyword_hit(ans: str, chunks, min_hits=1) -> int:
    keys = keywords_from_answer(ans)
    if not keys:
        return 0
    for ch in chunks:
        txt = norm(chunk_text_of(ch))
        hits = sum(1 for k in keys if k in txt)
        if hits >= min_hits:
            return 1
    return 0

def main():
    df = pd.read_csv(QA_CSV)
    df["question_text"] = df["question_text"].astype(str)
    df["answer_text"] = df["answer_text"].fillna("").astype(str)

    conn = get_connection()
    cur = conn.cursor()

    rows = []
    for _, r in df.iterrows():
        q = r["question_text"]
        a = r["answer_text"]
        chunks = retrieve_chunks(cur, q, top_k=K_RETRIEVE)[:K_EVAL]  # IMPORTANT
        hit = keyword_hit(a, chunks, min_hits=1)
        rows.append({"question_text": q, "answer_text": a, f"keyword_hit@{K_EVAL}": hit})

    cur.close()
    conn.close()

    out = pd.DataFrame(rows)
    score = out[f"keyword_hit@{K_EVAL}"].mean()

    out_csv = OUT / "keyword_eval_results.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n✅ Keyword-based Retrieval Metric")
    print(f"KeywordHit@{K_EVAL}: {score:.3f}")
    print(f"✅ Saved: {out_csv}")

if __name__ == "__main__":
    main()