# Explainable AI-Based Ransomware Detection System

Offline **tabular** prototype for a B.Tech CSE (AI & ML) demonstration. It trains Random Forest, XGBoost, or SVM on a CSV feature schema, reports a leakage-aware local split, and explains predictions with **SHAP** and **LIME**.

This is the software counterpart of the student papers:

- *AI-Based Ransomware Detection: A Review of Machine Learning, Explainable AI, Datasets, and Detection Challenges*
- *An Explainable AI-Based Ransomware Detection System: A Proposed Framework*

The papers state that the framework was **not experimentally validated**. Scores shown in this app are **computed on the data you load**. They are not copied from Alsaif (2023), Aljabri et al. (2024), or any other cited work.

## What this software does not do

- It does not execute uploaded files or scan binaries.
- It is not real-time endpoint protection or pre-encryption detection.
- The bundled CSV is **SYNTHETIC_DEMO** data, not ransomware telemetry.

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- About 1 GB disk for packages (SHAP / XGBoost)

## Installation

```powershell
cd C:\Users\Radhika\Documents\xai-ransomware-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\generate_demo_dataset.py
```

On macOS/Linux, replace the activate line with `source .venv/bin/activate`.

## Run the dashboard

From the project root:

```powershell
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

Recommended demo path:

1. **Dataset Management** → Load bundled synthetic demo CSV  
2. **Model Training** → Random Forest → Train  
3. **Ransomware Detection** → Predict selected row  
4. **Explainable AI** → inspect SHAP and LIME  
5. **Model Evaluation** → metrics and confusion matrix (labelled as local split / demo)

## Project layout

```
app.py                          Streamlit entry point
src/config.py                   Schema, paths, limits
src/security.py                 Upload validation (CSV only)
src/feature_extraction.py       Column validation
src/data_preprocessing.py       Cleaning, encoding, stratified split
src/model_training.py           RF / XGBoost / SVM + joblib
src/model_evaluation.py         Metrics (no invented scores)
src/prediction.py               Class + probability
src/shap_explainer.py           Global / local SHAP
src/lime_explainer.py           Local LIME
src/database.py                 SQLite history
src/visualization.py            Plotly charts
src/pipeline.py                 Module orchestration
data/sample_synthetic_demo.csv  Generated demo table
docs/                           Architecture, dataset format, training, tests, limits
tests/test_modules.py
```

## Documentation

| File | Contents |
|------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module communication and diagram |
| [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md) | Required CSV columns |
| [docs/TRAINING.md](docs/TRAINING.md) | How training is run |
| [docs/TESTING.md](docs/TESTING.md) | Tests and manual checks |
| [docs/LIMITATIONS_AND_FUTURE_WORK.md](docs/LIMITATIONS_AND_FUTURE_WORK.md) | Honest limits and next steps |

## Tests

```powershell
pytest -q
```
