from pathlib import Path
import requests
import pandas as pd

URL = ("https://raw.githubusercontent.com/Himanshu-1703/"
       "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "reddit.csv"


def download():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement depuis {URL} ...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    OUT.write_bytes(r.content)
    print(f"Sauvegardé dans {OUT}")

    df = pd.read_csv(OUT)
    print("\nColonnes :", list(df.columns))
    print(f"Nombre de commentaires : {len(df)}")
    label_col = "category" if "category" in df.columns else "label"
    print("\nDistribution des labels :")
    print(df[label_col].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    download()