# Architecture

## How modules communicate

Modules never scrape each other’s internals. `app.py` talks to `src/pipeline.py`. The pipeline calls the others through functions and dataclasses.

```
                    ┌──────────────────┐
                    │     app.py       │
                    │  Streamlit UI    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  src/pipeline.py │
                    └────────┬─────────┘
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
 security.py          feature_extraction.py   data_preprocessing.py
 (CSV only)           (schema check)          (impute, encode, split)
                             │
                             ▼
                    model_training.py ──► joblib files
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     model_evaluation   prediction.py   shap / lime
              │              │              │
              ▼              ▼              ▼
        visualization    database.py    Plotly charts
```

### Stage mapping (papers → code)

| Paper stage | Module |
|-------------|--------|
| Dataset selection / audit | `feature_extraction.py`, Dataset Management page |
| Preprocessing | `data_preprocessing.py` |
| Leakage-aware split | `leakage_aware_split()` — stratified random after exact-duplicate drop. Family-disjoint / temporal splits are **not** implemented (metadata is optional and the demo is not a capture). |
| Train RF / XGBoost / SVM | `model_training.py` |
| Evaluate | `model_evaluation.py` |
| Predict | `prediction.py` |
| SHAP + LIME | `shap_explainer.py`, `lime_explainer.py` |
| History | `database.py` (SQLite, no passwords) |

### Data contracts

- **Feature table:** pandas `DataFrame` with the columns in `src/config.py` (`FEATURE_COLUMNS`) plus optional metadata and `label`.
- **Trained model:** sklearn `Pipeline` (`preprocess` + `model`) stored by joblib.
- **Metrics:** dict produced only by `evaluate_classifier` from `y_true` / `y_pred` / `predict_proba`.
- **Explanations:** SHAP value arrays and LIME weight tables derived from that same pipeline. If an explainer fails, the UI shows the error instead of a fake chart.

### Integrity rules

1. Unknown CSV columns are ignored, not silently used.
2. Missing required features block train/predict.
3. Preprocessors fit on **training** rows only.
4. Test metrics are computed once per training click on the held-out split.
5. Rows with `data_source=SYNTHETIC_DEMO` are labelled as demonstration data in the UI.
