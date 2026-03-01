from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parent.parent / "data" / "router_train.csv"
OUT  = Path(__file__).resolve().parent.parent / "models" / "router_classifier.joblib"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Use fine-tuned embedding model ────────────
BASE       = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "rusl_embedding_model"

# Fallback to pre-trained if fine-tuned model doesn't exist yet
if MODEL_PATH.exists():
    MODEL_NAME = str(MODEL_PATH)
    print(f"✅ Using fine-tuned embedding model: {MODEL_NAME}")
else:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"⚠️  Fine-tuned model not found. Using pre-trained: {MODEL_NAME}")

def main():
    df = pd.read_csv(DATA)
    df["question_text"] = df["question_text"].astype(str).str.strip()
    df["label"]         = df["label"].astype(str).str.strip()

    X = df["question_text"].tolist()
    y = df["label"].tolist()

    print("✅ Loaded:", len(df))
    print(df["label"].value_counts())

    print(f"📌 Loading embedder: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    print("📌 Creating embeddings...")
    X_vec = embedder.encode(X, batch_size=32, show_progress_bar=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n📊 Classification Report")
    print(classification_report(y_test, y_pred))

    print("📌 Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    # Save model WITH the embed_model path so chat_service loads same model
    joblib.dump(
        {"classifier": clf, "embed_model": MODEL_NAME},
        OUT
    )

    print(f"\n✅ Saved model: {OUT}")
    print(f"   Embedding model used: {MODEL_NAME}")

if __name__ == "__main__":
    main()