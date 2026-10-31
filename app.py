"""Streamlit dashboard for the explainable ransomware detection prototype."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    CLASS_NAMES,
    DATA_DIR,
    DEMO_BANNER,
    FEATURE_COLUMNS,
    FEATURE_HELP,
    SUPPORTED_MODELS,
    ensure_runtime_dirs,
)
from src.database import fetch_history, history_counts
from src.data_preprocessing import class_balance, encode_labels
from src.feature_extraction import extract_feature_frame, feature_group_coverage, inspect_columns
from src.model_training import load_persisted_model, model_status
from src.pipeline import detect_and_explain, read_csv_bytes, train_and_evaluate
from src.visualization import (
    class_balance_pie,
    confusion_heatmap,
    lime_bar,
    metric_bars,
    shap_bar,
)

ensure_runtime_dirs()
st.set_page_config(
    page_title="Explainable AI Ransomware Detection",
    page_icon="🛡️",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
    h1, h2, h3 {letter-spacing: 0.02em;}
    .status-ok {background:#12352A;border:1px solid #3DDC97;padding:0.7rem 1rem;border-radius:8px;}
    .status-warn {background:#3A2A12;border:1px solid #E3B341;padding:0.7rem 1rem;border-radius:8px;}
    .status-bad {background:#3A1518;border:1px solid #E85D4C;padding:0.7rem 1rem;border-radius:8px;}
    .muted {color:#9AA8BE;font-size:0.92rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SAMPLE_CSV = DATA_DIR / "sample_synthetic_demo.csv"
PAGES = (
    "Dashboard",
    "Dataset Management",
    "Model Training",
    "Ransomware Detection",
    "Explainable AI",
    "Model Evaluation",
    "Prediction History",
    "About Project",
)


def _init_state() -> None:
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("dataset_name", None)
    st.session_state.setdefault("dataset_report", None)
    st.session_state.setdefault("train_bundle", None)
    st.session_state.setdefault("last_detection", None)
    st.session_state.setdefault("background_X", None)


def load_sample() -> pd.DataFrame:
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Bundled demo CSV not found at {SAMPLE_CSV}. Run scripts/generate_demo_dataset.py."
        )
    return pd.read_csv(SAMPLE_CSV)


def active_pipeline():
    bundle = st.session_state.get("train_bundle")
    if bundle is not None:
        return bundle["trained"].pipeline, bundle["trained"].name, bundle["is_synthetic_demo"]
    loaded = load_persisted_model()
    if loaded:
        return loaded["pipeline"], loaded["name"], False
    return None, None, False


def sidebar() -> str:
    st.sidebar.title("XAI Detection")
    st.sidebar.caption("Offline tabular prototype · not an endpoint scanner")
    page = st.sidebar.radio("Navigation", PAGES, index=0)
    status = model_status()
    if status["ready"]:
        st.sidebar.markdown(
            f"<div class='status-ok'>Model ready: <b>{status['name']}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div class='status-warn'>No trained model yet. Open Model Training.</div>",
            unsafe_allow_html=True,
        )
    if st.session_state.get("dataset_name"):
        st.sidebar.caption(f"Loaded table: {st.session_state['dataset_name']}")
    return page


def page_dashboard() -> None:
    st.title("Explainable AI-Based Ransomware Detection")
    st.markdown(
        "<p class='muted'>Classifies <b>tabular feature rows</b> as benign or ransomware-related "
        "and explains the model with SHAP and LIME. It does not execute files or perform live malware scanning.</p>",
        unsafe_allow_html=True,
    )
    status = model_status()
    hist = history_counts()
    ds = st.session_state.get("dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model status", "Ready" if status["ready"] else "Not trained")
    c2.metric("Active algorithm", status.get("name") or "—")
    c3.metric("History records", hist["total"])
    c4.metric("Stored ransomware labels", hist["ransomware_alerts"])
    if ds is not None:
        report = st.session_state.get("dataset_report")
        st.subheader("Loaded dataset")
        if report and report.is_synthetic_demo:
            st.warning(DEMO_BANNER)
        st.write(
            {
                "rows": int(len(ds)),
                "columns": int(ds.shape[1]),
                "source_name": st.session_state.get("dataset_name"),
            }
        )
    else:
        st.info("No dataset loaded. Use Dataset Management or the bundled synthetic demo CSV.")
    st.subheader("Pipeline (from the project papers)")
    st.markdown(
        """
1. **Tabular features** — static-like aggregates and behavioural counts (CSV only).
2. **Leakage-aware split** — drop exact duplicates; stratified train / validation / test.
3. **Models** — Random Forest, XGBoost, optional SVM; same split for all.
4. **Scores** — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, confusion matrix.
5. **Explanations** — SHAP (global + local) and LIME (local). Attributions are not causes.
        """
    )


def page_dataset() -> None:
    st.title("Dataset Management")
    st.caption("Upload a CSV feature table. Executables, dumps, and archives are rejected.")
    if SAMPLE_CSV.exists() and st.button("Load bundled synthetic demo CSV"):
        df = load_sample()
        table, report = extract_feature_frame(df, require_label=True)
        st.session_state.dataset = table
        st.session_state.dataset_name = SAMPLE_CSV.name
        st.session_state.dataset_report = report
        st.success(f"Loaded {len(table)} rows from {SAMPLE_CSV.name}.")
    uploaded = st.file_uploader("CSV feature table", type=["csv"])
    if uploaded is not None:
        try:
            df = read_csv_bytes(uploaded.name, uploaded.getvalue())
            table, report = extract_feature_frame(df, require_label=False)
            st.session_state.dataset = table
            st.session_state.dataset_name = uploaded.name
            st.session_state.dataset_report = report
            st.success(f"Accepted CSV '{uploaded.name}' ({len(table)} rows).")
        except Exception as exc:
            st.error(str(exc))
            return
    ds = st.session_state.get("dataset")
    if ds is None:
        st.stop()
    report = st.session_state.dataset_report
    if report.is_synthetic_demo:
        st.warning(DEMO_BANNER)
    for w in report.warnings:
        st.warning(w)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Schema check")
        st.json(
            {
                "available_features": report.available_features,
                "missing_required": report.missing_required,
                "ignored_columns": report.ignored_columns,
                "has_label": report.has_label,
                "can_train": report.can_train,
                "can_predict": report.can_predict,
                "group_coverage": feature_group_coverage(ds.columns),
            }
        )
    with c2:
        if report.has_label:
            y = encode_labels(ds["label"])
            bal = class_balance(y)
            st.plotly_chart(class_balance_pie(bal["n_benign"], bal["n_ransomware"]), use_container_width=True)
            st.caption("Counts are from the loaded table only.")
    st.subheader("Preview")
    st.dataframe(ds.head(25), use_container_width=True)
    st.subheader("Supported feature dictionary")
    st.dataframe(
        pd.DataFrame({"feature": FEATURE_COLUMNS, "meaning": [FEATURE_HELP[c] for c in FEATURE_COLUMNS]}),
        use_container_width=True,
        hide_index=True,
    )


def page_training() -> None:
    st.title("Model Training")
    ds = st.session_state.get("dataset")
    if ds is None:
        st.warning("Load a labelled CSV first.")
        st.stop()
    report = inspect_columns(ds.columns)
    if not report.has_label:
        st.error("This table has no label column. Training is disabled.")
        st.stop()
    if st.session_state.dataset_report and st.session_state.dataset_report.is_synthetic_demo:
        st.warning(DEMO_BANNER)
    model_name = st.selectbox("Algorithm", SUPPORTED_MODELS, index=0)
    st.caption("Hyperparameters are documented defaults. They are not tuned on the test set.")
    if st.button("Train on current table", type="primary"):
        with st.spinner("Training and evaluating on a stratified hold-out split…"):
            try:
                bundle = train_and_evaluate(ds, model_name)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        st.session_state.train_bundle = bundle
        st.session_state.background_X = bundle["split"].X_train
        st.success(
            f"Trained {model_name} in {bundle['trained'].training_seconds:.2f}s "
            f"(train={bundle['trained'].n_train}, valid={bundle['trained'].n_valid}, "
            f"test={bundle['trained'].n_test})."
        )
    bundle = st.session_state.get("train_bundle")
    if bundle is None:
        st.stop()
    st.subheader("Split notes")
    for note in bundle["split"].notes:
        st.write("•", note)
    st.subheader("Validation metrics (used for model inspection)")
    st.json({k: v for k, v in bundle["validation_metrics"].items() if k != "confusion_matrix"})
    st.subheader("Held-out test metrics (single use of this split)")
    st.json({k: v for k, v in bundle["test_metrics"].items() if k != "confusion_matrix"})
    st.caption(bundle["test_metrics"]["disclaimer"])


def page_detection() -> None:
    st.title("Ransomware Detection")
    st.caption("Scores feature rows. This is not real-time malware scanning of binaries.")
    pipeline, model_name, persisted_demo = active_pipeline()
    if pipeline is None:
        st.warning("Train a model first.")
        st.stop()
    ds = st.session_state.get("dataset")
    source = st.radio("Input", ("Use loaded table", "Upload a prediction CSV"))
    frame = None
    source_name = st.session_state.get("dataset_name") or "session"
    is_synthetic = bool(st.session_state.get("dataset_report") and st.session_state.dataset_report.is_synthetic_demo)
    if source == "Use loaded table":
        if ds is None:
            st.warning("Load a dataset first.")
            st.stop()
        frame, report = extract_feature_frame(ds, require_label=False)
        is_synthetic = report.is_synthetic_demo
    else:
        uploaded = st.file_uploader("Prediction CSV", type=["csv"], key="pred_csv")
        if uploaded is None:
            st.stop()
        try:
            raw = read_csv_bytes(uploaded.name, uploaded.getvalue())
            frame, report = extract_feature_frame(raw, require_label=False)
            source_name = uploaded.name
            is_synthetic = report.is_synthetic_demo
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    if is_synthetic:
        st.warning(DEMO_BANNER)
    idx = st.number_input("Row to score and explain", min_value=0, max_value=max(len(frame) - 1, 0), value=0, step=1)
    background = st.session_state.get("background_X")
    if background is None:
        background = frame.head(min(80, len(frame)))
    if st.button("Predict selected row", type="primary"):
        with st.spinner("Scoring and computing explanations…"):
            result = detect_and_explain(
                pipeline,
                model_name,
                frame,
                background,
                source_name=source_name,
                is_synthetic=is_synthetic,
                row_index=int(idx),
            )
        st.session_state.last_detection = result
    result = st.session_state.get("last_detection")
    if result is None:
        st.stop()
    selected = result["selected"]
    badge = "status-bad" if selected["predicted_class"] == "ransomware" else "status-ok"
    st.markdown(
        f"<div class='{badge}'><b>Predicted class:</b> {selected['predicted_class']}</div>",
        unsafe_allow_html=True,
    )
    st.write(selected.get("score_label"))
    c1, c2 = st.columns(2)
    c1.metric("P(benign)", _fmt_prob(selected.get("p_benign")))
    c2.metric("P(ransomware) — model score", _fmt_prob(selected.get("p_ransomware")))
    st.info(result["analyst_summary"])
    st.subheader("All scored rows")
    st.dataframe(result["predictions"], use_container_width=True)


def _fmt_prob(value) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.3f}"


def page_xai() -> None:
    st.title("Explainable AI")
    result = st.session_state.get("last_detection")
    if result is None:
        st.warning("Run a prediction first so explanations attach to a real model output.")
        st.stop()
    st.write(result["analyst_summary"])
    st.subheader("SHAP")
    if result["shap_error"]:
        st.error(result["shap_error"])
    elif result["shap"] is None:
        st.warning("SHAP output is not available.")
    else:
        st.caption(result["shap"]["method_note"])
        st.caption(f"Explanation time: {result['shap']['seconds']:.2f}s")
        local = result["shap"]["local"]
        st.plotly_chart(
            shap_bar(local.rename(columns={"abs_shap": "mean_abs_shap"})[["feature", "mean_abs_shap"]], "Local |SHAP| for the selected row"),
            use_container_width=True,
        )
        st.dataframe(local, use_container_width=True, hide_index=True)
        if "global_on_batch" in result["shap"]:
            st.plotly_chart(
                shap_bar(result["shap"]["global_on_batch"], "Mean |SHAP| on the scored batch (≤80 rows)"),
                use_container_width=True,
            )
    st.subheader("LIME")
    if result["lime_error"]:
        st.error(result["lime_error"])
    elif result["lime"] is None:
        st.warning("LIME output is not available.")
    else:
        st.caption(result["lime"]["method_note"])
        st.caption(f"Explanation time: {result['lime']['seconds']:.2f}s")
        st.plotly_chart(lime_bar(result["lime"]["weights"]), use_container_width=True)
        st.dataframe(result["lime"]["weights"], use_container_width=True, hide_index=True)


def page_evaluation() -> None:
    st.title("Model Evaluation")
    bundle = st.session_state.get("train_bundle")
    if bundle is None:
        loaded = load_persisted_model()
        if loaded is None:
            st.warning("Train a model in this session to see the confusion matrix and metric charts.")
            st.stop()
        from src.config import METRICS_PATH
        import json

        if not METRICS_PATH.exists():
            st.warning("No evaluation JSON is stored yet.")
            st.stop()
        stored = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.warning("Showing last persisted evaluation. Retrain in this session for live charts from the split objects.")
        st.json(stored)
        st.stop()
    if bundle["is_synthetic_demo"]:
        st.warning(DEMO_BANNER)
    test = bundle["test_metrics"]
    st.caption(test["disclaimer"])
    st.plotly_chart(metric_bars(test), use_container_width=True)
    st.plotly_chart(confusion_heatmap(test["confusion_matrix"]), use_container_width=True)
    fpr = test["false_positive_rate"]
    st.write(
        {
            "split": test["split"],
            "n_samples": test["n_samples"],
            "false_positive_rate": fpr if fpr is not None else "undefined (no negatives)",
            "inference_seconds": test["inference_seconds"],
            "source": test["source"],
        }
    )
    st.subheader("Validation (not the final reported test numbers)")
    st.plotly_chart(metric_bars(bundle["validation_metrics"]), use_container_width=True)


def page_history() -> None:
    st.title("Prediction History")
    frame = fetch_history()
    if frame.empty:
        st.info("No stored predictions yet.")
        st.stop()
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("is_synthetic = 1 means the source table declared SYNTHETIC_DEMO.")


def page_about() -> None:
    st.title("About this project")
    st.markdown(
        """
This software implements the **student framework** described in:

- *AI-Based Ransomware Detection: A Review of Machine Learning, Explainable AI, Datasets, and Detection Challenges*
- *An Explainable AI-Based Ransomware Detection System: A Proposed Framework*

Authors (papers): Parth Mavi, Mukul, Tushar Singh, Tanishk Garg. Guide: Dr. Shubajit Ghosh,
Meerut Institute of Engineering and Technology.

### What it is
A **modular, offline** prototype that trains Random Forest / XGBoost / SVM on a **CSV feature schema**,
then explains predictions with SHAP and LIME.

### What it is not
- Not a replacement for endpoint protection or antivirus.
- Not a real-time pre-encryption detector.
- Not an executor of malware samples.
- Not a claim of the accuracies reported by Alsaif (2023), Aljabri et al. (2024), or other cited papers.
  Those figures used different tasks and datasets and **must not be compared** with this app’s local scores.

### Architecture
`security` → `feature_extraction` → `data_preprocessing` → `model_training` → `model_evaluation`
→ `prediction` → `shap_explainer` / `lime_explainer` → `database` / `visualization` → `app.py`.

See `docs/ARCHITECTURE.md` and `docs/LIMITATIONS_AND_FUTURE_WORK.md`.
        """
    )


_init_state()
page = sidebar()
{
    "Dashboard": page_dashboard,
    "Dataset Management": page_dataset,
    "Model Training": page_training,
    "Ransomware Detection": page_detection,
    "Explainable AI": page_xai,
    "Model Evaluation": page_evaluation,
    "Prediction History": page_history,
    "About Project": page_about,
}[page]()
