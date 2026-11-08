"""Train and evaluate the five required Dry Bean classifiers.

Run from the project root with: python train.py
The UCI data is retrieved with ucimlrepo unless a local CSV is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
TEST_DATA_PATH = ROOT / "test_data.csv"
RANDOM_STATE = 42
TARGET = "Class"


def load_dataset(csv_path: Path | None) -> tuple[pd.DataFrame, pd.Series]:
    """Load a local CSV or fetch UCI dataset 602 via the official client."""
    if csv_path:
        frame = pd.read_csv(csv_path)
        if TARGET not in frame.columns:
            raise ValueError(f"The input CSV must include a '{TARGET}' target column.")
        return frame.drop(columns=[TARGET]), frame[TARGET].astype(str)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Install project dependencies with `pip install -r requirements.txt`, "
            "or provide --data-csv PATH to a downloaded UCI Dry Bean CSV."
        ) from exc

    dataset = fetch_ucirepo(id=602)
    features = dataset.data.features.copy()
    target_frame = dataset.data.targets
    target = target_frame.iloc[:, 0].astype(str).rename(TARGET)
    return features, target


def build_models() -> dict[str, object]:
    """Return comparable model pipelines with scaling where it is needed."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, min_samples_leaf=2
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=3.0, kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def calculate_metrics(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC (OvR macro)": roc_auc_score(
            y_test, y_prob, labels=model.classes_, multi_class="ovr", average="macro"
        ),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, help="Optional local CSV with Class target column")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    X, y = load_dataset(args.data_csv)
    if len(X.columns) < 12 or len(X) < 500:
        raise ValueError("The selected data does not meet the assignment's 12-feature/500-row minimum.")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values found. Resolve them using training-only preprocessing first.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All input features must be numeric for this project.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    # Keep the held-out target in the CSV: the app evaluates uploaded labeled rows.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test
    test_frame.to_csv(TEST_DATA_PATH, index=False)

    metric_rows: list[dict[str, object]] = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test)
        file_name = name.lower().replace("-", "_").replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / file_name)
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        pd.DataFrame(confusion_matrix(y_test, y_pred, labels=model.classes_),
                     index=model.classes_, columns=model.classes_).to_csv(
                         RESULTS_DIR / f"{file_name.removesuffix('.joblib')}_confusion_matrix.csv"
                     )
        with (RESULTS_DIR / f"{file_name.removesuffix('.joblib')}_classification_report.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, indent=2)
        metric_rows.append({"Model": name, **metrics})
        print(f"{name}: " + " | ".join(f"{key}={value:.4f}" for key, value in metrics.items()))

    pd.DataFrame(metric_rows).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    metadata = {
        "dataset": "UCI Dry Bean (dataset id 602)",
        "target": TARGET,
        "feature_columns": X.columns.tolist(),
        "classes": sorted(y.unique().tolist()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": args.test_size,
        "random_state": RANDOM_STATE,
        "metrics": "Accuracy; macro one-vs-rest AUC; macro precision/recall/F1; multiclass MCC",
    }
    (MODEL_DIR / "model_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved models to {MODEL_DIR}, results to {RESULTS_DIR}, and test rows to {TEST_DATA_PATH}.")


if __name__ == "__main__":
    main()
