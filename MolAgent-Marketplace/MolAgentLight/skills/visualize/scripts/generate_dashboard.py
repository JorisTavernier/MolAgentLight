# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.0",
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "scipy>=1.10",
# ]
# ///
"""Generate an interactive HTML dashboard from AutoMol evaluation results.

Reads ``pipeline_state.json`` and the per-property evaluation CSVs, computes
all derived metrics (residuals, ROC/PR, calibration, hit enrichment, etc.) in
Python, then substitutes that pre-computed JSON blob into the playground
template at ``${MOLAGENT_PLUGIN_ROOT}/playgrounds/dashboard_playground.html``.

The same template file is what NEXUS loads as an iframe playground — Nexus
just substitutes the same ``{{INITIAL_DATA}}`` placeholder with its own data
payload at iframe-bake time. One source of truth, two delivery paths.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

PLACEHOLDER_DATA = "{{INITIAL_DATA}}"
PLACEHOLDER_TITLE = "{{ARTIFACT_TITLE}}"
PLACEHOLDER_RUN_ID = "{{RUN_ID}}"
PLACEHOLDER_ARTIFACT_ID = "{{ARTIFACT_ID}}"


def resolve_template_path() -> Path:
    """Find ``playgrounds/dashboard_playground.html`` regardless of how the
    script is invoked. Order:
      1. ``MOLAGENT_PLUGIN_ROOT`` env (set by the SessionStart hook)
      2. Walk up from this script's location to the plugin root
    """
    plugin_root = os.environ.get("MOLAGENT_PLUGIN_ROOT")
    if plugin_root:
        candidate = Path(plugin_root) / "playgrounds" / "dashboard_playground.html"
        if candidate.exists():
            return candidate
    # Fallback: scripts/ → visualize/ → skills/ → plugin root
    here = Path(__file__).resolve()
    for parent in [here.parents[3], here.parents[2], here.parents[1]]:
        candidate = parent / "playgrounds" / "dashboard_playground.html"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate playgrounds/dashboard_playground.html. "
        "Set MOLAGENT_PLUGIN_ROOT or restore the playground template."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average with shrinking window at edges."""
    if len(a) < 2 * (n - 1) + 1:
        n = max(1, (len(a) - 1) // 2)
    if n <= 1:
        return a.copy()
    ret = np.cumsum(a, dtype=float)
    reverse = np.cumsum(a[::-1], dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    reverse[n:] = reverse[n:] - reverse[:-n]
    ret[n - 1 : -n + 1] = ret[n - 1 : -n + 1] - a[n - 1 : -n + 1]
    ret[n - 1 : -n + 1] = (ret[n - 1 : -n + 1] + reverse[-n : n - 2 : -1]) / (
        2 * (n - 1) + 1
    )
    for i in range(n - 1):
        ret[i] = reverse[-1 - i]
        for j in range(i):
            ret[i] = ret[i] + a[j]
            ret[-1 - i] = ret[-1 - i] + a[-1 - j]
        ret[i] = ret[i] / (n + i)
        ret[-1 - i] = ret[-1 - i] / (n + i)
    return ret


def load_pipeline_state(state_path: Path) -> dict:
    with open(state_path) as f:
        return json.load(f)


def load_evaluation_csv(csv_path: str, run_folder: Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.is_absolute():
        candidate = run_folder.parent.parent / p
        if candidate.exists():
            p = candidate
    if not p.exists():
        p = run_folder / Path(csv_path).name
    return pd.read_csv(p)


def load_train_info(train_info_path: str, run_folder: Path) -> dict | None:
    p = Path(train_info_path)
    if not p.is_absolute():
        candidate = run_folder.parent.parent / p
        if candidate.exists():
            p = candidate
    if not p.exists():
        p = run_folder / Path(train_info_path).name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Compute regression-derived data
# ---------------------------------------------------------------------------

def compute_regression_data(
    df: pd.DataFrame, prop: str, metrics_from_state: dict
) -> dict:
    true_col = f"true_{prop}"
    pred_col = f"predicted_{prop}"
    sd_col = f"SD_{prop}"

    y_true = df[true_col].values.astype(float)
    y_pred = df[pred_col].values.astype(float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    smiles = (
        df.loc[mask, "Stand_SMILES"].tolist()
        if "Stand_SMILES" in df.columns
        else []
    )

    has_sd = sd_col in df.columns
    sd_vals = (
        df.loc[mask, sd_col].values.astype(float).tolist() if has_sd else []
    )

    n = len(y_true)
    abs_errors = np.abs(y_true - y_pred)

    # Metrics
    if n >= 2:
        try:
            pearson_r = pearsonr(y_true, y_pred).statistic
        except Exception:
            pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = None

    metrics = {
        "n": n,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if n >= 2 else None,
        "pearson": pearson_r,
        **(metrics_from_state or {}),
    }

    return {
        "true": y_true.tolist(),
        "predicted": y_pred.tolist(),
        "sd": sd_vals,
        "smiles": smiles,
        "n_samples": int(n),
        "metrics": metrics,
        "pearson": pearson_r,
    }


# ---------------------------------------------------------------------------
# Compute classification-derived data
# ---------------------------------------------------------------------------

def compute_classification_data(
    df: pd.DataFrame, prop: str, metrics_from_state: dict
) -> dict:
    true_cols = [f"true_Class_{prop}", f"true_{prop}"]
    pred_cols = [f"predicted_Class_{prop}", f"predicted_{prop}"]

    true_col = next((c for c in true_cols if c in df.columns), None)
    pred_col = next((c for c in pred_cols if c in df.columns), None)
    if true_col is None or pred_col is None:
        return {"error": f"Missing true/pred columns for {prop}", "n_samples": 0, "metrics": {}}

    y_true = df[true_col].values
    y_pred = df[pred_col].values
    mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    smiles = (
        df.loc[mask, "Stand_SMILES"].tolist()
        if "Stand_SMILES" in df.columns
        else []
    )

    # Confusion matrix
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Probability columns: prob_<class> for each class
    prob_cols = {c: f"prob_{c}" for c in classes if f"prob_{c}" in df.columns}
    roc_curves: dict = {}
    pr_curves: dict = {}
    calibration: dict = {}

    metrics = {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        **(metrics_from_state or {}),
    }

    # ROC + PR + calibration when probabilities available
    if prob_cols:
        for cls, col in prob_cols.items():
            probs = df.loc[mask, col].values.astype(float)
            y_bin = (y_true == cls).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            fpr, tpr, _ = roc_curve(y_bin, probs)
            try:
                a = float(auc(fpr, tpr))
            except Exception:
                a = None
            roc_curves[str(cls)] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": a}

            prec, rec, _ = precision_recall_curve(y_bin, probs)
            pr_curves[str(cls)] = {"precision": prec.tolist(), "recall": rec.tolist()}

            # Reliability/calibration: bin into deciles
            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            mp = []  # mean predicted
            fp = []  # fraction positive
            for b in range(n_bins):
                in_bin = (probs >= bins[b]) & (probs < bins[b + 1])
                if in_bin.sum() == 0:
                    continue
                mp.append(float(probs[in_bin].mean()))
                fp.append(float(y_bin[in_bin].mean()))
            calibration[str(cls)] = {
                "mean_predicted": mp,
                "fraction_positive": fp,
            }

        # Macro AUC
        try:
            metrics["auc"] = float(np.mean([v["auc"] for v in roc_curves.values() if v.get("auc") is not None]))
        except Exception:
            pass

    return {
        "true": [str(v) for v in y_true.tolist()],
        "predicted": [str(v) for v in y_pred.tolist()],
        "smiles": smiles,
        "n_samples": int(mask.sum()),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "class_labels": [str(c) for c in classes],
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# Render: substitute placeholders in the playground template
# ---------------------------------------------------------------------------

def sanitize_for_json(obj):
    """Replace NaN/Inf/numpy scalars with JSON-safe equivalents."""
    # Booleans first — numpy.bool_ inherits from int, so isinstance(np.bool_, int)
    # is True; without an explicit branch it'd silently coerce to 0/1.
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def render_dashboard(
    run_id: str,
    task_type: str,
    properties: list[str],
    property_data: dict,
    train_info_per_prop: dict,
    config: dict,
    output_path: Path,
    title: str | None = None,
    artifact_id: str | None = None,
):
    template_path = resolve_template_path()
    template = template_path.read_text(encoding="utf-8")

    payload = {
        "_task_type": task_type,
        "_run_id": run_id,
        "_title": title or f"MolAgent — {run_id}",
        "_properties": properties,
        "_train_info": train_info_per_prop,
        "_config": config,
    }
    payload.update(property_data)
    payload = sanitize_for_json(payload)
    data_json = json.dumps(payload, default=str, ensure_ascii=False)

    # Escape closing-script tag inside JSON to keep <script> parsing intact
    data_json = data_json.replace("</", "<\\/")

    html = (
        template
        .replace(PLACEHOLDER_DATA, data_json)
        .replace(PLACEHOLDER_TITLE, title or f"MolAgent — {run_id}")
        .replace(PLACEHOLDER_RUN_ID, run_id)
        .replace(PLACEHOLDER_ARTIFACT_ID, artifact_id or run_id)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--pipeline-state",
    required=True,
    type=click.Path(exists=True),
    help="Path to pipeline_state.json",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output HTML file path",
)
@click.option(
    "--title",
    default=None,
    help="Dashboard title (defaults to 'MolAgent — {run_id}')",
)
@click.option("--verbose", is_flag=True, help="Print progress info")
def main(pipeline_state: str, output: str, title: str | None, verbose: bool):
    """Generate an interactive dashboard from AutoMol evaluation results."""
    state_path = Path(pipeline_state).resolve()
    output_path = Path(output).resolve()
    run_folder = state_path.parent

    if verbose:
        click.echo(f"Loading pipeline state from {state_path}")

    state = load_pipeline_state(state_path)
    run_id = state.get("run_id", "unknown")
    config = state.get("config", {})
    task_type = config.get("task_type", "regression")
    target_properties = config.get("target_properties", [])
    outputs = state.get("outputs", state.get("files", {}))
    eval_results = outputs.get("evaluation_results", {})
    train_info_paths = outputs.get("train_info", {})
    metrics_from_state = state.get("metrics", {})

    if not eval_results:
        click.echo("ERROR: No evaluation results found in pipeline state.", err=True)
        click.echo("Run the evaluate step (step 5) first.", err=True)
        sys.exit(1)

    if verbose:
        click.echo(f"Run: {run_id}")
        click.echo(f"Task type: {task_type}")
        click.echo(f"Properties: {target_properties}")

    property_data: dict = {}
    train_info_per_prop: dict = {}
    for prop in target_properties:
        csv_path = eval_results.get(prop)
        if not csv_path:
            if verbose:
                click.echo(f"  Skipping {prop}: no evaluation CSV")
            continue

        if verbose:
            click.echo(f"  Loading evaluation CSV for {prop}: {csv_path}")

        df = load_evaluation_csv(csv_path, run_folder)
        if verbose:
            click.echo(f"    {len(df)} rows, columns: {list(df.columns)}")

        prop_metrics = metrics_from_state.get(prop, {})
        ti_path = train_info_paths.get(prop)
        ti = load_train_info(ti_path, run_folder) if ti_path else None
        if ti:
            train_info_per_prop[prop] = ti

        if task_type == "regression":
            property_data[prop] = compute_regression_data(df, prop, prop_metrics)
        elif task_type == "classification":
            property_data[prop] = compute_classification_data(df, prop, prop_metrics)
        else:
            click.echo(
                f"  WARNING: Unknown task type '{task_type}', treating as regression"
            )
            property_data[prop] = compute_regression_data(df, prop, prop_metrics)

    if not property_data:
        click.echo("ERROR: No valid evaluation data found for any property.", err=True)
        sys.exit(1)

    if verbose:
        click.echo(f"Rendering dashboard to {output_path}")

    render_dashboard(
        run_id=run_id,
        task_type=task_type,
        properties=list(property_data.keys()),
        property_data=property_data,
        train_info_per_prop=train_info_per_prop,
        config=config,
        output_path=output_path,
        title=title,
    )

    if verbose:
        click.echo(f"Dashboard written: {output_path}")


if __name__ == "__main__":
    main()
