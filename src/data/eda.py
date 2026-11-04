from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "reddit_clean.csv"
FIG = ROOT / "reports" / "figures"
NAMES = {-1: "Négatif", 0: "Neutre", 1: "Positif"}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["n_words"] = df["text"].str.split().str.len()
    df["sentiment"] = df["label"].map(NAMES)

    # 1. Distribution des classes
    plt.figure(figsize=(6, 4))
    order = list(NAMES.values())
    ax = sns.countplot(data=df, x="sentiment", order=order)
    ax.bar_label(ax.containers[0])
    plt.title("Distribution des classes")
    plt.tight_layout()
    plt.savefig(FIG / "class_distribution.png", dpi=150)
    plt.close()

    # 2. Longueur des textes par classe
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="sentiment", y="n_words", order=order, showfliers=False)
    plt.title("Longueur des commentaires (mots) par classe")
    plt.tight_layout()
    plt.savefig(FIG / "text_length.png", dpi=150)
    plt.close()

    print("Proportion des classes :")
    print(df["sentiment"].value_counts(normalize=True).round(3))
    print("\nLongueur (mots) :")
    print(df["n_words"].describe().round(1))
    print(f"\nFigures sauvegardées dans {FIG}")


if __name__ == "__main__":
    main()