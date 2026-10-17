from pathlib import Path

import pandas as pd
from datasets import load_dataset

SEED = 42
THRESHOLD = 0.5
N_TRAIN = 200_000
N_VAL = 20_000

LABELS = ["toxic", "obscene", "threat", "insult", "identity_hate"]
SRC = {
    "toxic": "toxicity",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_hate": "identity_attack",
}


def to_frame(split) -> pd.DataFrame:
    df = split.to_pandas()
    out = pd.DataFrame({"comment_text": df["text"].fillna("").str.strip()})
    for new, old in SRC.items():
        out[new] = (df[old] >= THRESHOLD).astype(int)
    return out[out["comment_text"].str.len() > 0].reset_index(drop=True)


def build(out_dir: str = "data/processed") -> None:
    ds = load_dataset("google/civil_comments")
    train = to_frame(ds["train"].shuffle(seed=SEED).select(range(N_TRAIN)))
    val = to_frame(ds["validation"].shuffle(seed=SEED).select(range(N_VAL)))
    test = to_frame(ds["test"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame.to_parquet(out / f"{name}.parquet", index=False)
        print(f"\n{name}: {len(frame):,} rows")
        print("positives per label:\n", frame[LABELS].sum().to_string())
        print("clean rate:", round((~frame[LABELS].any(axis=1)).mean(), 3))


if __name__ == "__main__":
    build()
