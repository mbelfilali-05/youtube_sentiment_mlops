import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
RANDOM_STATE = 42


def load(name):
    df = pd.read_csv(PROCESSED / name)
    df["text"] = df["text"].fillna("")
    return df["text"], df["label"]


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                           max_features=20000, sublinear_tf=True)


def compare_models(X_train, y_train):
    """Cross-validation on TRAIN only (the test set stays untouched)."""
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "ComplementNB": ComplementNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
    }
    results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", clf)])
        t0 = time.time()
        scores = cross_val_score(pipe, X_train, y_train, cv=3,
                                 scoring="f1_macro")
        results[name] = round(float(scores.mean()), 4)
        print(f"{name:20s} F1-macro CV = {scores.mean():.4f} "
              f"({time.time() - t0:.0f}s)")
    return results


def tune_logreg(X_train, y_train):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [20000, 50000],
        "tfidf__min_df": [1, 2],
        "clf__C": [1, 5, 10],
    }
    search = GridSearchCV(pipe, grid, cv=3, scoring="f1_macro",
                          n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print("\nBest params:", search.best_params_)
    print(f"Best F1-macro (CV): {search.best_score_:.4f}")
    return search


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load("train.csv")
    X_test, y_test = load("test.csv")

    print("=== 1. Algorithm comparison (3-fold CV on train) ===")
    cv_results = compare_models(X_train, y_train)

    print("\n=== 2. GridSearchCV on Logistic Regression ===")
    search = tune_logreg(X_train, y_train)
    best = search.best_estimator_

    print("\n=== 3. Final evaluation on the test set ===")
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Neutral", "Positive"],
        output_dict=True)
    print(classification_report(
        y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Negative", "Neutral", "Positive"],
        cmap="Blues")
    disp.ax_.set_title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrix.png", dpi=150)
    plt.close()

    print("=== 4. Inference time (batch of 50) ===")
    batch = list(X_test.iloc[:50])
    vec, clf = best.named_steps["tfidf"], best.named_steps["clf"]
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        clf.predict(vec.transform(batch))
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(f"Average: {avg_ms:.2f} ms")

    # Save vectorizer and model separately (TP requirement)
    joblib.dump(vec, MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(clf, MODELS / "sentiment_model.joblib")

    f1_per_class = {k: round(report[k]["f1-score"], 4)
                    for k in ["Negative", "Neutral", "Positive"]}
    metrics = {
        "cv_comparison_f1_macro": cv_results,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_per_class": f1_per_class,
        "inference_ms_batch50": round(avg_ms, 2),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== TP criteria check ===")
    print(f"Accuracy >= 0.80      : {acc:.4f} -> {'OK' if acc >= 0.80 else 'FAIL'}")
    for k, v in f1_per_class.items():
        print(f"F1 {k:9s} > 0.75 : {v:.4f} -> {'OK' if v > 0.75 else 'FAIL'}")
    print(f"Inference < 100 ms    : {avg_ms:.2f} -> {'OK' if avg_ms < 100 else 'FAIL'}")


if __name__ == "__main__":
    main()