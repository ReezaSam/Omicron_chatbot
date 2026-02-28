# =============================================================
# evaluate_ml_router.py
# Evaluate trained ML router on eval_questions table
# =============================================================

from pathlib import Path
import os
import joblib
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
# Load .env
# ─────────────────────────────────────────────
HERE = Path(__file__).resolve()
env_path = HERE.parent.parent / ".env"
load_dotenv(env_path)

DB_NAME = os.getenv("DB_NAME", "omicron_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

MODEL_PATH = HERE.parent.parent / "models" / "router_classifier.joblib"

# ─────────────────────────────────────────────
# DB Connection
# ─────────────────────────────────────────────
def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("📌 Loading ML router model...")
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["classifier"]
    embed_name = bundle["embed_model"]

    print("📌 Loading embedding model...")
    embedder = SentenceTransformer(embed_name)

    print("📌 Fetching evaluation questions...")
    conn = connect()
    df = pd.read_sql(
        "SELECT question_text, action_expected FROM eval_questions ORDER BY eval_id;",
        conn
    )
    conn.close()

    X = df["question_text"].astype(str).tolist()
    y_true = df["action_expected"].astype(str).str.lower().tolist()

    print("📌 Creating embeddings for eval questions...")
    X_vec = embedder.encode(X, batch_size=32, show_progress_bar=True)

    y_pred = clf.predict(X_vec)

    print("\n📊 ML ROUTER RESULTS (on eval_questions)")
    print("-------------------------------------------------")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=3))

    print("Confusion Matrix (answer, clarify, escalate):")
    labels = ["answer", "clarify", "escalate"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)

if __name__ == "__main__":
    main()