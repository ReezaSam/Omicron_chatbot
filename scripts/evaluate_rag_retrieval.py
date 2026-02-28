from pathlib import Path
import re
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from Rag_Retriever import get_connection, retrieve_chunks

HERE = Path(__file__).resolve()
BASE = HERE.parent.parent
load_dotenv(BASE / ".env")

QA_CSV = BASE / "data" / "qa_pairs.csv"
OUT_DIR = BASE / "scripts" / "eval_rag_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_EVAL = 5          # metric is Hit@5
K_RETRIEVE = 15     # retrieval depth to allow good candidates
THRESH = 0.20

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","was","were",
    "be","been","being","it","this","that","these","those","as","at","from","into","about"
}

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokenize(s: str):
    s = norm(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t and t not in STOPWORDS]

def overlap_score(answer_text: str, chunk_text: str) -> float:
    a = tokenize(answer_text)
    c = tokenize(chunk_text)
    if not a or not c:
        return 0.0
    set_a = set(a)
    set_c = set(c)
    inter = len(set_a & set_c)
    prec = inter / max(1, len(set_c))
    rec  = inter / max(1, len(set_a))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def chunk_text_of(ch):
    if isinstance(ch, tuple) and len(ch) >= 1:
        return ch[0] or ""
    if isinstance(ch, dict):
        return ch.get("chunk_text", "") or ch.get("text", "") or ""
    return ""

def best_rank_soft(answer_text: str, chunks, threshold: float) -> int:
    for i, ch in enumerate(chunks, start=1):
        sc = overlap_score(answer_text, chunk_text_of(ch))
        if sc >= threshold:
            return i
    return 0

def best_overlap_in_topk(answer_text: str, chunks) -> float:
    best = 0.0
    for ch in chunks:
        sc = overlap_score(answer_text, chunk_text_of(ch))
        best = max(best, sc)
    return best

def main():
    df = pd.read_csv(QA_CSV)
    df["question_text"] = df["question_text"].astype(str)
    df["answer_text"] = df["answer_text"].fillna("").astype(str)

    conn = get_connection()
    cur = conn.cursor()

    results = []
    print(f"📌 Evaluating retrieval on {len(df)} QA pairs (Retrieve={K_RETRIEVE}, Eval@{K_EVAL}, thresh={THRESH}) ...")

    for _, row in df.iterrows():
        q = row["question_text"]
        gold = row["answer_text"]

        chunks = retrieve_chunks(cur, q, top_k=K_RETRIEVE)[:K_EVAL]  # IMPORTANT

        rank = best_rank_soft(gold, chunks, threshold=THRESH)
        hit = 1 if rank > 0 else 0
        mrr = (1.0 / rank) if rank > 0 else 0.0

        results.append({
            "question_text": q,
            "answer_text": gold,
            f"hit@{K_EVAL}": hit,
            "first_rank": rank,
            "mrr": mrr,
            "best_overlap": best_overlap_in_topk(gold, chunks),
        })

    cur.close()
    conn.close()

    out_df = pd.DataFrame(results)
    hit_rate = float(out_df[f"hit@{K_EVAL}"].mean())
    mrr_mean = float(out_df["mrr"].mean())

    print("\n✅ Retrieval Metrics (SOFT MATCH)")
    print(f"Recall@{K_EVAL} (Hit@{K_EVAL}): {hit_rate:.3f}")
    print(f"MRR@{K_EVAL}: {mrr_mean:.3f}")
    print(f"Average best_overlap@{K_EVAL}: {out_df['best_overlap'].mean():.3f}")

    csv_path = OUT_DIR / "retrieval_eval_results.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ Saved: {csv_path}")

    plt.figure()
    out_df["best_overlap"].dropna().plot(kind="hist", bins=30)
    plt.title(f"Best Overlap@{K_EVAL} Distribution")
    plt.xlabel("Overlap score")
    plt.ylabel("Count")
    p1 = OUT_DIR / "best_overlap_hist.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {p1}")

if __name__ == "__main__":
    main()