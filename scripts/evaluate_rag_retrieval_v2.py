# =============================================================
# evaluate_rag_retrieval_v2.py
# Omicron UniBot — RAG Retrieval Evaluation (Upgraded Metrics)
#
# Metrics added vs v1:
#   ✅ ExactAnswerHit@K  — gold answer string found exactly in chunk
#   ✅ NumericHit@K      — digits/IDs from gold answer found in chunk
#   ✅ KeywordHit@K      — keyword overlap (same as v1)
#   ✅ SoftHit@K         — soft overlap (same as original)
#
# Output: eval_rag_outputs_v2/full_eval_results.csv + summary print
# =============================================================

from pathlib import Path
import re
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from Rag_Retriever import get_connection, retrieve_chunks

HERE = Path(__file__).resolve()
BASE = HERE.parent.parent
load_dotenv(BASE / ".env")

QA_CSV   = BASE / "data" / "qa_pairs.csv"
OUT_DIR  = BASE / "scripts" / "eval_rag_outputs_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_EVAL     = 10    # evaluate Hit@5
K_RETRIEVE = 15   # retrieve more, then trim to K_EVAL

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","was","were",
    "be","been","being","it","this","that","these","those","as","at","from","into","about"
}
MONTHS = {
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"
}

# ─────────────────────────────────────────────
# TEXT UTILS
# ─────────────────────────────────────────────

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokenize(s: str):
    s = norm(s)
    s = re.sub(r"[^a-z0-9/\s]", " ", s)   # keep slash for "896/2" style
    return [t for t in s.split() if t and t not in STOPWORDS]

def chunk_text_of(ch) -> str:
    if isinstance(ch, tuple) and len(ch) >= 1:
        return ch[0] or ""
    if isinstance(ch, dict):
        return ch.get("chunk_text", "") or ch.get("text", "") or ""
    return ""

# ─────────────────────────────────────────────
# METRIC 1: ExactAnswerHit@K
# Does the gold answer string appear literally inside any top-K chunk?
# Great for short facts: "1998", "Anuradhapura", "896/2"
# ─────────────────────────────────────────────

def exact_answer_hit(gold: str, chunks) -> int:
    gold_norm = norm(gold)
    if not gold_norm:
        return 0
    for ch in chunks:
        if gold_norm in norm(chunk_text_of(ch)):
            return 1
    return 0

# ─────────────────────────────────────────────
# METRIC 2: NumericHit@K
# Extract all numbers/IDs from gold answer.
# Check if ANY retrieved chunk contains ALL of them.
# Great for: "896/2", "1998", "3.0", student ID formats
# ─────────────────────────────────────────────

def extract_numerics(text: str):
    """Extract digit sequences and slash-separated IDs like 896/2"""
    text = norm(text)
    # match patterns like 896/2, 1998, 3.0, etc.
    return re.findall(r"\d+(?:[./]\d+)*", text)

def numeric_hit(gold: str, chunks) -> int:
    nums = extract_numerics(gold)
    if not nums:
        return 0   # no numeric content in gold → skip this metric
    for ch in chunks:
        txt = norm(chunk_text_of(ch))
        if all(n in txt for n in nums):
            return 1
    return 0

def has_numerics(gold: str) -> bool:
    return len(extract_numerics(gold)) > 0

# ─────────────────────────────────────────────
# METRIC 3: KeywordHit@K (same as v1)
# Keywords = digits, month names, long words, or slash tokens from answer
# ─────────────────────────────────────────────

def keywords_from_answer(ans: str):
    toks = tokenize(ans)
    keys = []
    for t in toks:
        if t.isdigit() or t in MONTHS or "/" in t or len(t) >= 6:
            keys.append(t)
    return list(dict.fromkeys(keys)) if keys else list(dict.fromkeys(toks))

def keyword_hit(gold: str, chunks, min_hits: int = 1) -> int:
    keys = keywords_from_answer(gold)
    if not keys:
        return 0
    for ch in chunks:
        txt = norm(chunk_text_of(ch))
        hits = sum(1 for k in keys if k in txt)
        if hits >= min_hits:
            return 1
    return 0

# ─────────────────────────────────────────────
# METRIC 4: SoftHit@K (same as original v0)
# Token overlap F1 score >= threshold
# ─────────────────────────────────────────────

SOFT_THRESH = 0.20

def overlap_score(answer_text: str, chunk_text: str) -> float:
    a = set(tokenize(answer_text))
    c = set(tokenize(chunk_text))
    if not a or not c:
        return 0.0
    inter = len(a & c)
    prec  = inter / max(1, len(c))
    rec   = inter / max(1, len(a))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def soft_hit(gold: str, chunks, threshold: float = SOFT_THRESH) -> int:
    for ch in chunks:
        if overlap_score(gold, chunk_text_of(ch)) >= threshold:
            return 1
    return 0

# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

def main():
    df = pd.read_csv(QA_CSV)
    df["question_text"] = df["question_text"].astype(str)
    df["answer_text"]   = df["answer_text"].fillna("").astype(str)

    conn = get_connection()
    cur  = conn.cursor()

    rows = []
    numeric_total = 0

    print(f"📌 Evaluating {len(df)} QA pairs | Retrieve={K_RETRIEVE} | Eval@{K_EVAL}")
    print("─" * 60)

    for _, r in df.iterrows():
        q    = r["question_text"]
        gold = r["answer_text"]

        chunks = retrieve_chunks(cur, q, top_k=K_RETRIEVE)[:K_EVAL]

        exact = exact_answer_hit(gold, chunks)
        num   = numeric_hit(gold, chunks)
        kw    = keyword_hit(gold, chunks)
        soft  = soft_hit(gold, chunks)
        has_n = has_numerics(gold)

        if has_n:
            numeric_total += 1

        rows.append({
            "question_text":          q,
            "answer_text":            gold,
            f"exact_hit@{K_EVAL}":    exact,
            f"numeric_hit@{K_EVAL}":  num if has_n else None,  # None = not applicable
            f"keyword_hit@{K_EVAL}":  kw,
            f"soft_hit@{K_EVAL}":     soft,
            "has_numeric_answer":     has_n,
        })

    cur.close()
    conn.close()

    out_df = pd.DataFrame(rows)

    # ── Summary ──────────────────────────────
    exact_rate   = out_df[f"exact_hit@{K_EVAL}"].mean()
    keyword_rate = out_df[f"keyword_hit@{K_EVAL}"].mean()
    soft_rate    = out_df[f"soft_hit@{K_EVAL}"].mean()

    # NumericHit only over rows where answer has numbers
    num_df       = out_df[out_df["has_numeric_answer"] == True]
    numeric_rate = num_df[f"numeric_hit@{K_EVAL}"].mean() if len(num_df) > 0 else 0.0

    print("\n✅ RETRIEVAL EVALUATION RESULTS")
    print("=" * 45)
    print(f"  ExactAnswerHit@{K_EVAL}  : {exact_rate:.3f}   ← NEW")
    print(f"  NumericHit@{K_EVAL}      : {numeric_rate:.3f}   ← NEW (on {len(num_df)} numeric answers)")
    print(f"  KeywordHit@{K_EVAL}      : {keyword_rate:.3f}")
    print(f"  SoftHit@{K_EVAL}         : {soft_rate:.3f}")
    print("=" * 45)
    print(f"  Total QA pairs evaluated : {len(out_df)}")
    print(f"  Answers with numbers     : {numeric_total}")

    # ── Save CSV ─────────────────────────────
    csv_path = OUT_DIR / "full_eval_results.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n✅ Saved full results: {csv_path}")

    # ── Plot ─────────────────────────────────
    metrics = {
        f"ExactHit@{K_EVAL}":   exact_rate,
        f"NumericHit@{K_EVAL}": numeric_rate,
        f"KeywordHit@{K_EVAL}": keyword_rate,
        f"SoftHit@{K_EVAL}":    soft_rate,
    }
    plt.figure(figsize=(8, 4))
    bars = plt.bar(metrics.keys(), metrics.values(), color=["#2ecc71","#3498db","#e67e22","#9b59b6"])
    plt.ylim(0, 1.0)
    plt.ylabel("Hit Rate")
    plt.title(f"RAG Retrieval Metrics @ K={K_EVAL}")
    for bar, val in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    chart_path = OUT_DIR / "retrieval_metrics_bar.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved chart: {chart_path}")

if __name__ == "__main__":
    main()