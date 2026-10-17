from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42


def make_split(raw_path="data/raw/train.csv", out_dir="data/processed", val_size=0.1):
    df = pd.read_csv(raw_path)
    # Light cleanup only: keep text as-is for the transformer tokenizer.
    df["comment_text"] = df["comment_text"].fillna("").str.strip()

    strat = df[LABELS].any(axis=1)
    train, val = train_test_split(
        df, test_size=val_size, random_state=SEED, stratify=strat
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)

    print(f"train: {len(train):,}  val: {len(val):,}")
    print("train label rates:\n", train[LABELS].mean().round(4))
    print("val label rates:\n", val[LABELS].mean().round(4))


if __name__ == "__main__":
    make_split()