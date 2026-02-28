from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "data"

ANS = BASE / "answer_questions_500.csv"
CE  = BASE / "clarify_escalate_400.csv"
OUT = BASE / "router_train.csv"

def main():
    if not ANS.exists():
        raise FileNotFoundError(f"Missing file: {ANS}")
    if not CE.exists():
        raise FileNotFoundError(f"Missing file: {CE}")

    df1 = pd.read_csv(ANS)

    # ✅ robust CSV read for clarify_escalate
    try:
        df2 = pd.read_csv(CE, engine="python", on_bad_lines="warn")
    except TypeError:
        # for older pandas
        df2 = pd.read_csv(CE, engine="python", error_bad_lines=False, warn_bad_lines=True)

    # Ensure correct columns
    if "question_text" not in df2.columns or "label" not in df2.columns:
        raise ValueError(f"{CE} must have columns: question_text,label")

    # clean
    for df in (df1, df2):
        df["question_text"] = df["question_text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = pd.concat([df1, df2], ignore_index=True)

    # keep only valid labels
    df = df[df["label"].isin(["answer", "clarify", "escalate"])]

    # remove duplicates
    df = df.drop_duplicates(subset=["question_text"]).reset_index(drop=True)

    # shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print("✅ router_train.csv created successfully!")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()