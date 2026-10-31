# Model training instructions

## In the UI

1. Load a labelled CSV (or the bundled demo).
2. Open **Model Training**.
3. Choose `random_forest`, `xgboost`, or `svm`.
4. Click **Train on current table**.

The run:

1. Validates the schema.
2. Coerces numeric columns; median imputation is **fit on train only**.
3. Drops exact duplicate feature+label rows.
4. Stratified split: 64% train, 16% validation, 20% test (0.8 × 0.2 valid inside the 80% train pool).
5. Fits the chosen estimator.
6. Writes `models/active_model.joblib` and `models/last_evaluation.json`.

SVM uses `StandardScaler` in the same train-fitted pipeline. Tree models do not scale features.

XGBoost uses `compute_sample_weight(class_weight="balanced")` on the **training fold only**. Random Forest and SVM use `class_weight="balanced"`.

## Hyperparameters

Defaults live in `src/model_training.py`. They are fixed starting points, not a nested CV search. The papers list grouped / nested CV as a planned experiment (E2); that search is **not** implemented here.

## Interpreting scores

- **Validation** metrics are for inspecting the fit.
- **Test** metrics are a single hold-out estimate for that local table and seed (`RANDOM_STATE=42`).
- If `data_source` contains `SYNTHETIC_DEMO`, the UI labels the source `synthetic_demo_split`.
- Do not compare these numbers to published accuracies on other datasets.

## Retraining

Each successful train overwrites `models/active_model.joblib`. Previous joblib files are not versioned.
