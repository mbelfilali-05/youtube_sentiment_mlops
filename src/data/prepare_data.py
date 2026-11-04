import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.text_cleaning import clean_text

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "reddit.csv"
PROCESSED = ROOT / "data" / "processed"

TEST_SIZE = 0.2
RANDOM_STATE = 42



def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    rename = {"clean_comment": "text", "category": "label"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df[["text", "label"]]


def main():
    df = load_raw()
    print(f"Brut : {len(df)} lignes")

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text")
    print(f"Après nettoyage : {len(df)} lignes")

    # Vérification des spécifications du TP
    counts = df["label"].value_counts()
    assert len(df) >= 1000, "Dataset trop petit (< 1000)"
    assert (counts >= 300).all(), f"Une classe a moins de 300 exemples :\n{counts}"

    # Split reproductible et stratifié (même proportion de classes)
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "reddit_clean.csv", index=False)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)

    print(f"\nTrain : {len(train)} | Test : {len(test)}")
    print("Distribution train :\n", train["label"].value_counts(normalize=True).sort_index().round(3))
    print("Distribution test  :\n", test["label"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    main()