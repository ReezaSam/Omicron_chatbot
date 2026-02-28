from pathlib import Path
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses
)
from torch.utils.data import DataLoader
from dotenv import load_dotenv

HERE = Path(__file__).resolve()
BASE = HERE.parent.parent
load_dotenv(BASE / ".env")

QA_CSV     = BASE / "data" / "qa_pairs.csv"
OUTPUT_DIR = BASE / "models" / "rusl_embedding_model"

# ── Config ────────────────────────────────
EPOCHS     = 3
BATCH_SIZE = 16
WARMUP     = 50

def main():
    # Load your QA pairs
    df = pd.read_csv(QA_CSV)
    df["question_text"] = df["question_text"].astype(str)
    df["answer_text"]   = df["answer_text"].fillna("").astype(str)

    # Create training examples
    # Each example = (question, answer) pair
    # The model learns: question and answer should be CLOSE in vector space
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(
            texts=[row["question_text"], row["answer_text"]]
        ))

    print(f"✅ Training examples: {len(examples)}")

    # Load base model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # DataLoader
    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    # Loss function
    # MultipleNegativesRankingLoss:
    # Makes (question, correct_answer) close
    # and pushes other answers in batch away
    loss = losses.MultipleNegativesRankingLoss(model)

    # Fine-tune
    print(f"📌 Fine-tuning for {EPOCHS} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP,
        show_progress_bar=True,
        output_path=str(OUTPUT_DIR)
    )

    print(f"✅ Fine-tuned model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()