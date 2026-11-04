import json
import time
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def load(name):
    df = pd.read_csv(PROCESSED / name)
    df["text"] = df["text"].fillna("")
    return df["text"], df["label"]


def main():
    X_train, y_train = load("train.csv")
    X_test, y_test = load("test.csv")

    features = FeatureUnion([
        ("word", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                 min_df=2, max_features=30000)),
        ("char", TfidfVectorizer(sublinear_tf=True, analyzer="char_wb",
                                 ngram_range=(2, 5), min_df=3,
                                 max_features=60000)),
    ])
    pipe = Pipeline([
        ("tfidf", features),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    grid = {
        "tfidf__word__ngram_range": [(1, 1), (1, 2)],
        "tfidf__char__ngram_range": [(2, 4), (3, 5)],
        "clf__C": [2, 5, 10, 20],
    }
    search = GridSearchCV(pipe, grid, cv=3, scoring="f1_macro",
                          n_jobs=-1, verbose=1)
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"\nSearch done in {time.time() - t0:.0f}s")
    print("Best params:", search.best_params_)
    print(f"Best F1-macro (CV): {search.best_score_:.4f}  (previous: 0.8106)")

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    names = ["Negative", "Neutral", "Positive"]
    print(classification_report(y_test, y_pred, target_names=names))
    report = classification_report(y_test, y_pred, target_names=names,
                                   output_dict=True)
    f1_neg = report["Negative"]["f1-score"]
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} | F1 Negative: {f1_neg:.4f}")

    # Inference time (full pipeline, batch of 50)
    batch = list(X_test.iloc[:50])
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        best.predict(batch)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(f"Inference batch 50: {avg_ms:.2f} ms")

    if search.best_score_ > 0.8106:
        # Save the full pipeline (features + classifier) in one file
        joblib.dump(best, MODELS / "sentiment_pipeline.joblib")
        metrics = {
            "model": "FeatureUnion(word+char TF-IDF) + LogisticRegression",
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "cv_f1_macro": round(search.best_score_, 4),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "f1_per_class": {k: round(report[k]["f1-score"], 4) for k in names},
            "inference_ms_batch50": round(avg_ms, 2),
        }
        (REPORTS / "metrics_v2.json").write_text(json.dumps(metrics, indent=2))
        print("\nSaved models/sentiment_pipeline.joblib + reports/metrics_v2.json")
    else:
        print("\nNo CV improvement: nothing saved.")


if __name__ == "__main__":
    main()