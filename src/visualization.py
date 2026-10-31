"""Plotly charts for metrics and explanations. Values must come from callers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_LAYOUT = dict(
    paper_bgcolor="#0B1220",
    plot_bgcolor="#151D2E",
    font=dict(color="#E8EEF7"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def confusion_heatmap(cm: dict) -> go.Figure:
    matrix = cm["matrix"]
    labels = cm["labels"]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"Pred {x}" for x in labels],
            y=[f"True {x}" for x in labels],
            colorscale="Teal",
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(title="Confusion matrix (this local split only)", **DARK_LAYOUT)
    return fig


def metric_bars(metrics: dict) -> go.Figure:
    names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    items = [(n, metrics.get(n)) for n in names if metrics.get(n) is not None]
    frame = pd.DataFrame(items, columns=["metric", "value"])
    fig = px.bar(frame, x="metric", y="value", range_y=[0, 1], title="Local evaluation scores")
    fig.update_layout(**DARK_LAYOUT)
    return fig


def shap_bar(importance: pd.DataFrame, title: str) -> go.Figure:
    data = importance.head(15).iloc[::-1]
    fig = px.bar(data, x=data.columns[1], y="feature", orientation="h", title=title)
    fig.update_layout(**DARK_LAYOUT, height=480)
    return fig


def lime_bar(weights: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        weights.iloc[::-1],
        x="weight",
        y="rule",
        orientation="h",
        title="LIME local weights (surrogate model)",
    )
    fig.update_layout(**DARK_LAYOUT, height=480)
    return fig


def class_balance_pie(n_benign: int, n_ransomware: int) -> go.Figure:
    fig = px.pie(
        names=["benign", "ransomware"],
        values=[n_benign, n_ransomware],
        title="Label counts in the loaded table",
        color_discrete_sequence=["#5B8DEF", "#E85D4C"],
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig
