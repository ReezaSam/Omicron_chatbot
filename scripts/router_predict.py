from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "router_classifier.joblib"

_bundle = joblib.load(MODEL_PATH)
clf = _bundle["classifier"]
embed_model_name = _bundle["embed_model"]

embedder = SentenceTransformer(embed_model_name)

def predict_action(question: str) -> str:
    vec = embedder.encode([question])[0]
    return clf.predict([vec])[0]