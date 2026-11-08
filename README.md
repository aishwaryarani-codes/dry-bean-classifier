# Dry Bean Variety Classifier

A machine learning project that compares five classifiers for identifying dry bean varieties from 16 numerical measurements. The Streamlit dashboard lets you evaluate labeled test data, compare model scores, and inspect class-level results.

**Repository:** [aishwaryarani-codes/dry-bean-classifier](https://github.com/aishwaryarani-codes/dry-bean-classifier/tree/feature/initialCode)  
**Live app:** [dry-bean-classifier-aishwaryarani.streamlit.app](https://dry-bean-classifier-aishwaryarani.streamlit.app/)

## a. Problem statement

Dry bean images can be summarized by geometric measurements such as area, perimeter, axis lengths, compactness, and shape factors. This project uses those measurements to predict one of seven bean varieties and compares five standard classification methods on the same held-out test set.

## b. Dataset description

The project uses the [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry%2Bbean) (dataset ID 602). It contains **13,611 instances**, **16 numeric features**, and the target label `Class`, with seven varieties: Barbunya, Bombay, Cali, Dermason, Horoz, Seker, and Sira. UCI reports no missing values.

The measurements were extracted from bean images using computer vision. This project uses the tabular measurements rather than the source images. The dataset is licensed under CC BY 4.0. Please credit the [UCI dataset](https://doi.org/10.24432/C50S4B) when reusing it.

## c. GitHub Repository Link

The repository contains the training and evaluation code, Streamlit app, dependency list, labeled held-out test CSV, saved model pipelines, and experiment results.

## d. Models used

The data is divided into a stratified 80% training set and 20% test set with `random_state=42` (10,888 training rows and 2,723 test rows). Scaling is included in pipelines for Logistic Regression, K-Nearest Neighbors, and Support Vector Machine so each scaler is fit using training data only.

Precision, recall, and F1 are macro averages. AUC is one-vs-rest macro using predicted probabilities. MCC is the multiclass Matthews correlation coefficient.

| Model | Accuracy | AUC | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.8884 | 0.9531 | 0.9038 | 0.9041 | 0.9036 | 0.8652 |
| K-Nearest Neighbors | 0.9166 | 0.9860 | 0.9318 | 0.9271 | 0.9292 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | **0.9229** | **0.9948** | **0.9360** | **0.9340** | **0.9350** | **0.9067** |

### Model observations

| Model | Observation |
|---|---|
| Logistic Regression | A strong linear baseline with 0.9207 accuracy and 0.9329 macro F1. Its MCC (0.9042) is close to the best model. |
| Decision Tree | Accuracy (0.8884) and macro F1 (0.9036) are lower than the other models except Naive Bayes. It is the most interpretable individual tree in this comparison. |
| K-Nearest Neighbors | Second-highest accuracy (0.9166) and macro F1 (0.9292). Scaling supports its distance-based comparisons. |
| Gaussian Naive Bayes | Lowest accuracy (0.7639), macro F1 (0.7677), and MCC (0.7154), though its AUC (0.9672) is higher than the Decision Tree's. |
| Support Vector Machine | Best accuracy (0.9229), macro precision (0.9360), macro recall (0.9340), macro F1 (0.9350), and MCC (0.9067); it ties Logistic Regression on AUC (0.9948). |
| **Overall winner** | **Support Vector Machine** leads on five of the six measures and ties for AUC on this held-out split. |

These figures describe this test split and may differ on new data.

## e. Streamlit app

**Live app:** [https://dry-bean-classifier-aishwaryarani.streamlit.app/](https://dry-bean-classifier-aishwaryarani.streamlit.app/)

The dashboard loads the saved models and does not retrain on uploaded data. It supports:

- CSV upload for labeled test records
- Selection of any of the five classifiers
- A leaderboard comparing every model on the same uploaded rows
- Accuracy, AUC, macro precision, recall, F1, and MCC
- Confusion matrix and per-class classification report

The uploaded CSV must contain all 16 feature columns and the `Class` target column. A sample labeled test file is included as `test_data.csv`.

## Run locally

1. Create and activate a Python 3.10 or newer environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the models and generate test data and results:

   ```bash
   python train.py
   ```

   By default, this fetches UCI dataset 602 using `ucimlrepo`. To train from a local CSV instead, pass a file with a `Class` target column:

   ```bash
   python train.py --data-csv path/to/dry_bean.csv
   ```

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Project structure

```text
.
├── app.py                         Streamlit dashboard
├── train.py                       Training and held-out evaluation
├── metrics.py                     Shared evaluation metrics
├── charts.py                      Confusion-matrix display
├── theme.py                       Theme and CSS helper
├── assets/style.css               Dashboard styling
├── .streamlit/config.toml         Streamlit theme settings
├── requirements.txt               Python dependencies
├── test_data.csv                  Labeled held-out test set
├── data/README.md                 Dataset source and attribution
├── model/                          Saved model pipelines and metadata
└── results/                        Metrics, confusion matrices, and reports
```

Training writes the five saved pipelines and `model/model_info.json` under `model/`, the held-out labeled rows to `test_data.csv`, and the comparison metrics, confusion matrices, and per-class reports under `results/`.
