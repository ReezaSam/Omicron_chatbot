import os
import re
from pathlib import Path

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ==============================
# ENV + DB CONFIG
# ==============================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv("DB_NAME", "omicron_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DOC_ID = int(os.getenv("DOC_ID", "1"))

# ==============================
# MODEL + RETRIEVAL CONFIG
# ==============================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
TOP_K = 5

DEFAULT_THRESHOLD = 0.57
ADMIN_THRESHOLD = 0.52
MIN_KEYWORD_OVERLAP = 1

OUT_DIR = os.path.join(os.path.dirname(__file__), "eval_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"📌 Loading embedding model: {EMBED_MODEL}")
embed_model = SentenceTransformer(EMBED_MODEL)

# ==============================
# PATTERNS
# ==============================
ACADEMIC_HINTS = {
    "gpa", "grade", "grading", "course", "subject", "semester", "year",
    "credit", "exam", "registration", "appeal", "suspension", "results",
    "repeat", "eligibility", "attendance", "programme", "degree",
    "defer", "withdraw", "transfer", "medical",
    "disciplinary", "misconduct", "deferment", "extension", "deadline",
    "transcript", "certificate", "graduation"
}

ADMIN_EVAL_PAT = re.compile(
    r"\b(appeal|appeals|suspend|suspension|disciplinary|misconduct|"
    r"re[-\s]?evaluation|recheck|remark|transfer|withdraw|"
    r"defer|deferment|special\s+consideration|medical|"
    r"extension|deadline|registration\s+issue|transcript|certificate|graduation)\b",
    re.I
)

STOPWORDS = {
    "the","a","an","is","are","was","were","to","of","in","on","for","and","or",
    "with","by","i","we","you","they","he","she","it","this","that","what","which",
    "when","where","how","can","could","should","do","does","did","me","my","your",
    "our","their"
}

# ✅ clarify if course-intent AND no department mentioned
COURSE_INTENT = re.compile(r"\b(subjects?|courses?|modules?|units?|electives?|core\s+subjects?)\b", re.I)
DEPT_PRESENT  = re.compile(
    r"\b(computing|physical\s+science|biological\s+science|health\s+promotion|chemical\s+sciences|ict|it|cs)\b",
    re.I
)

def should_clarify(q: str) -> bool:
    ql = q.lower()
    return bool(COURSE_INTENT.search(ql)) and not bool(DEPT_PRESENT.search(ql))

# ==============================
# HELPERS
# ==============================
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def looks_academic(msg: str) -> bool:
    m = normalize(msg)
    if any(k in m for k in ACADEMIC_HINTS):
        return True
    if re.search(r"\b(degree|programme|faculty|department|semester|exam|gpa|grade|course|subject)\b", m):
        return True
    return False

def tokenize(s: str):
    s = normalize(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t not in STOPWORDS and len(t) > 2]

def keyword_overlap(q: str, chunk: str) -> int:
    return len(set(tokenize(q)).intersection(set(tokenize(chunk))))

def to_pgvector(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

def connect():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )

def retrieve_topk(cur, q_vec):
    sql = f"""
        SELECT chunk_text, (embedding <=> %s::vector({DIM})) AS distance
        FROM kb_chunks
        WHERE doc_id=%s
        ORDER BY embedding <=> %s::vector({DIM})
        LIMIT %s;
    """
    cur.execute(sql, (q_vec, DOC_ID, q_vec, TOP_K))
    return cur.fetchall()

def combine_for_clarify(original_q: str, clarification_reply: str) -> str:
    return f"{original_q.rstrip('?')} for {clarification_reply}?"

# ==============================
# METRICS
# ==============================
def confusion_matrix(labels, preds, label_order):
    idx = {l: i for i, l in enumerate(label_order)}
    mat = np.zeros((len(label_order), len(label_order)), dtype=int)
    for y, p in zip(labels, preds):
        if y in idx and p in idx:
            mat[idx[y], idx[p]] += 1
    return mat

def precision_recall_f1(labels, preds, positive):
    tp = sum((y == positive and p == positive) for y, p in zip(labels, preds))
    fp = sum((y != positive and p == positive) for y, p in zip(labels, preds))
    fn = sum((y == positive and p != positive) for y, p in zip(labels, preds))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

# ==============================
# PREDICT (TURN-1)
# ==============================
def predict_action_turn1(cur, question: str):
    if not looks_academic(question):
        return "answer", None, None

    if should_clarify(question):
        return "clarify", None, None

    q_vec = to_pgvector(embed_model.encode(question).tolist())
    results = retrieve_topk(cur, q_vec)
    if not results:
        return "escalate", None, None

    best_text, best_dist = results[0][0], float(results[0][1])
    overlap = keyword_overlap(question, best_text)

    threshold = ADMIN_THRESHOLD if ADMIN_EVAL_PAT.search(question) else DEFAULT_THRESHOLD

    # distance-first escalation
    if best_dist > threshold:
        return "escalate", best_dist, overlap

    # soft zone
    if best_dist > (threshold - 0.05) and overlap < MIN_KEYWORD_OVERLAP:
        return "escalate", best_dist, overlap

    return "answer", best_dist, overlap

# ==============================
# MAIN
# ==============================
def main():
    conn = connect()
    cur  = conn.cursor()

    df = pd.read_sql("SELECT * FROM eval_questions ORDER BY eval_id;", conn)
    if df.empty:
        print("❌ eval_questions is empty.")
        cur.close(); conn.close()
        return

    # TURN-1 predictions for confusion matrix (routing accuracy)
    p1_list = []
    dist_list = []
    ov_list = []

    # TURN-2: clarify success
    p2_list = []
    clarify_success = []

    for _, row in df.iterrows():
        q = str(row["question_text"])
        expected = str(row["action_expected"]).lower().strip()

        p1, dist1, ov1 = predict_action_turn1(cur, q)
        p1_list.append(p1)
        dist_list.append(dist1 if dist1 is not None else np.nan)
        ov_list.append(ov1 if ov1 is not None else np.nan)

        # If it is a clarify case AND reply exists, run turn-2
        reply = row.get("clarification_reply", None)
        p2 = ""
        ok = ""
        if expected == "clarify" and reply is not None and str(reply).strip():
            q2 = combine_for_clarify(q, str(reply).strip())
            p2, _, _ = predict_action_turn1(cur, q2)

            # success definition: after clarify, it should NOT still be clarify
            ok = (p2 != "clarify")
        p2_list.append(p2)
        clarify_success.append(ok)

    df["pred_turn1"] = p1_list
    df["best_distance"] = dist_list
    df["keyword_overlap"] = ov_list
    df["pred_turn2_after_reply"] = p2_list
    df["clarify_success_after_reply"] = clarify_success

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ Saved CSV: {csv_path}")

    # Confusion matrix on TURN-1 routing
    y_true = df["action_expected"].astype(str).tolist()
    y_pred = df["pred_turn1"].astype(str).tolist()
    order  = ["answer", "clarify", "escalate"]

    cm = confusion_matrix(y_true, y_pred, order)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Turn-1 Routing)", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Expected", fontsize=12)
    plt.xticks(range(len(order)), order, rotation=25, fontsize=11)
    plt.yticks(range(len(order)), order, fontsize=11)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    thr = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(j, i, str(v), ha="center", va="center",
                     color=("white" if v > thr else "black"),
                     fontsize=12, fontweight="bold")

    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Saved confusion matrix: {cm_path}")

    # Metrics
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    print("\n📊 Per-class Metrics (Turn-1 Routing):")
    for cls in order:
        p, r, f = precision_recall_f1(y_true, y_pred, cls)
        print(f"  {cls:10s} → Precision: {p:.2f}  Recall: {r:.2f}  F1: {f:.2f}")

    # Clarify success rate
    clarify_rows = df[df["action_expected"].astype(str).str.lower() == "clarify"]
    valid = clarify_rows[clarify_rows["clarify_success_after_reply"].isin([True, False])]
    if len(valid) > 0:
        csr = float(valid["clarify_success_after_reply"].mean())
        print(f"\n✅ Clarification success rate (after reply): {csr:.3f}  (n={len(valid)})")
    else:
        csr = None
        print("\n⚠️ No clarify rows had clarification_reply filled, so cannot compute clarify success rate.")

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join([
            f"Total eval questions: {len(df)}",
            f"Turn-1 routing accuracy: {acc:.3f}",
            f"Clarify success rate after reply: {csr}",
            f"Thresholds: default={DEFAULT_THRESHOLD}, admin={ADMIN_THRESHOLD}",
            f"TopK: {TOP_K}",
            f"Output:",
            f"  - {csv_path}",
            f"  - {cm_path}",
        ]) + "\n")

    print(f"\n✅ Saved summary: {summary_path}")
    print(f"✅ Turn-1 routing accuracy: {acc:.3f}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()