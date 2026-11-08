"""Streamlit app for evaluating the saved Dry Bean classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from charts import show_confusion_matrix
from metrics import evaluate_classifier
from theme import apply_custom_css

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")
apply_custom_css()


@st.cache_resource
def load_artifacts():
    info_path = MODEL_DIR / "model_info.json"
    if not info_path.exists():
        raise FileNotFoundError("Trained model files are missing. Run `python train.py` first.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    models = {}
    absent = []
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
        else:
            absent.append(filename)
    if absent:
        raise FileNotFoundError("Missing saved model(s): " + ", ".join(absent) + ". Run `python train.py`.")
    return info, models


st.markdown(
    """
    <section class="hero">
      <div class="hero-copy">
        <div class="hero-kicker">A small study in shape, data &amp; prediction</div>
        <h1>Dry Bean <span>Classifier</span></h1>
        <p>Explore how five machine learning models distinguish seven bean varieties from their measured shapes.</p>
        <div class="hero-tags">
          <span>7 varieties</span><span>16 measurements</span><span>5 classifiers</span>
        </div>
      </div>
      <div class="hero-mark" aria-hidden="true">🫘</div>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    info, models = load_artifacts()
except (FileNotFoundError, json.JSONDecodeError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.markdown("### Your evaluation")
    st.caption("Choose a model and evaluate labeled bean measurements.")
    selected_model = st.selectbox("Choose a classifier", list(models))
    uploaded = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "Download sample test CSV",
            data=sample_path.read_bytes(),
            file_name="sample_test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.caption("The CSV must contain all 16 feature columns and the `Class` target column.")

if uploaded is None:
    default_path = ROOT / "test_data.csv"
    if default_path.exists():
        data = pd.read_csv(default_path)
        st.info("Using the bundled held-out test set. Upload a labeled CSV to explore another sample.")
    else:
        st.warning("Upload a labeled test CSV to begin.")
        st.stop()
else:
    try:
        data = pd.read_csv(uploaded)
    except Exception as error:  # Provide an actionable message for malformed user CSVs.
        st.error(f"Could not read this CSV: {error}")
        st.stop()

feature_columns = info["feature_columns"]
target_column = info["target"]
required = feature_columns + [target_column]
missing = [column for column in required if column not in data.columns]
if missing:
    st.error("Missing required column(s): " + ", ".join(missing))
    st.stop()
if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
if data[required].isna().any().any():
    st.error("Required feature and target columns must not contain missing values.")
    st.stop()
try:
    X = data[feature_columns].apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError):
    st.error("All feature columns must contain numeric values.")
    st.stop()

y = data[target_column].astype(str)
known_classes = set(info["classes"])
unknown = sorted(set(y) - known_classes)
if unknown:
    st.error("Unknown target class label(s): " + ", ".join(unknown))
    st.stop()

model_results = {}
for model_name, candidate in models.items():
    try:
        model_results[model_name] = evaluate_classifier(candidate, X, y)
    except ValueError as error:
        st.error(
            "Could not calculate multiclass AUC for this upload. Include examples from all seven "
            f"bean varieties and check the target labels. Details: {error}"
        )
        st.stop()

results, matrix, report, labels = model_results[selected_model]

st.markdown(
    f'<div class="section-heading"><div><div class="eyebrow">Selected model</div>'
    f'<h2>{selected_model}</h2></div><div class="result-chip">{len(data):,} rows evaluated</div></div>',
    unsafe_allow_html=True,
)
metric_columns = st.columns(6)
for column, (name, value) in zip(metric_columns, results.items()):
    column.metric(name, f"{value:.4f}")

st.markdown('<div class="section-label">THE FULL PICTURE</div>', unsafe_allow_html=True)
st.subheader("Compare all five models")
st.caption("Every model is scored on the same uploaded rows, so their results are directly comparable.")
leaderboard = pd.DataFrame(
    {name: score[0] for name, score in model_results.items()}
).T.rename_axis("Model").reset_index()
leaderboard = leaderboard.sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
metric_names = list(results)
leaderboard_view = leaderboard.style.format({metric: "{:.4f}" for metric in metric_names})
leaderboard_view = leaderboard_view.background_gradient(
    subset=["Accuracy", "F1 (macro)", "MCC"], cmap="GnBu", vmin=0, vmax=1
)
st.dataframe(leaderboard_view, hide_index=True, use_container_width=True)

st.markdown('<div class="section-label">MODEL DETAILS</div>', unsafe_allow_html=True)
left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("Confusion matrix")
        st.caption("Rows show the true variety; columns show the predicted variety.")
        show_confusion_matrix(matrix, labels)
with right:
    with st.container(border=True):
        st.subheader("Per-class report")
        st.caption("Precision, recall, and F1 for each bean variety.")
        report_frame = pd.DataFrame(report).T
        st.dataframe(report_frame.round(4), use_container_width=True)

st.markdown(
    '<div class="footer-note">Macro averages give each of the seven bean varieties equal weight.</div>',
    unsafe_allow_html=True,
)
