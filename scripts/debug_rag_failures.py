from pathlib import Path
import re
import pandas as pd
from dotenv import load_dotenv

import Rag_Retriever
from Rag_Retriever import get_connection, retrieve_chunks

HERE = Path(__file__).resolve()
BASE = HERE.parent.parent
load_dotenv(BASE / ".env")

QA_CSV = BASE / "data" / "qa_pairs.csv"

K_EVAL = 5
K_RETRIEVE = 15
THRESH = 0.20

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","was","were",
    "be","been","being","it","this","that","these","those","as","at","from","into","about"
}

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokenize(s: str):
    s = norm(s)
    s = re.sub(r"[^a-z0-9/\s]", " ", s)
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

def main():
    print("Rag_Retriever loaded from:", Rag_Retriever.__file__)
    print("Rag_Retriever TOP_K:", getattr(Rag_Retriever, "TOP_K", None))

    df = pd.read_csv(QA_CSV)
    df["question_text"] = df["question_text"].astype(str)
    df["answer_text"] = df["answer_text"].fillna("").astype(str)

    conn = get_connection()
    cur = conn.cursor()

    shown = 0
    for _, row in df.iterrows():
        q = row["question_text"]
        gold = row["answer_text"]

        chunks_all = retrieve_chunks(cur, q, top_k=K_RETRIEVE)
        chunks = chunks_all[:K_EVAL]

        best = 0.0
        best_i = -1
        scores = []

        for i, ch in enumerate(chunks, start=1):
            txt = chunk_text_of(ch)
            sc = overlap_score(gold, txt)
            scores.append((i, sc, txt[:350].replace("\n", " ")))
            if sc > best:
                best = sc
                best_i = i

        if best < THRESH:
            print("\n" + "="*80)
            print("Q:", q)
            print("Gold A:", gold)
            print(f"Best overlap@{K_EVAL}={best:.3f} (at rank {best_i})")
            print(f"- Top{K_EVAL}:")
            for i, sc, preview in scores:
                print(f"  {i}) overlap={sc:.3f}  preview={preview}")
            shown += 1
            if shown >= 10:
                break

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()