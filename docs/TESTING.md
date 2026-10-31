# Testing instructions

## Automated

From the project root, with the virtualenv activated:

```powershell
pytest -q
```

Covered behaviour:

- Rejection of non-CSV uploads
- Schema inspection (unknown columns ignored)
- Label encoding
- Stratified split + Random Forest fit on a tiny synthetic table
- Prediction frame columns

## Manual demonstration checklist

1. `streamlit run app.py` starts without import errors.
2. Loading the demo CSV shows the SYNTHETIC_DEMO banner.
3. Training Random Forest completes and shows validation **and** test dicts with a disclaimer.
4. Predicting a row shows predicted class, `P(benign)`, and `P(ransomware)` labelled as model probability.
5. Explainable AI page shows SHAP and LIME from that prediction, or a real error if an explainer fails.
6. Evaluation page confusion matrix matches the printed TP/FP/TN/FN.
7. Uploading `something.exe` is rejected.
8. About page states the software is not an endpoint scanner.

## What tests do not claim

Tests do **not** assert a minimum accuracy. Any accuracy on the tiny random table is incidental.
