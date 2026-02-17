#!/usr/bin/env python3
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from graph_builder import create_simple_dataset
from GNN_architectures import create_gnn_model
from training_utils import evaluate
from training_utils import compute_min_positive_prob, evaluate_threshold_from_train
from load_features import autotune_resources

# Plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import os

# Optional plotting deps
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None

# =====================================================
# Utilities
# =====================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, filename: str, plot_dir: Path):
    ensure_dir(plot_dir)
    fig.savefig(plot_dir / filename, dpi=200, bbox_inches="tight")


# =====================================================
# Prediction utilities (deterministic + MC Dropout)
# =====================================================

def _predict_probs_and_labels(model, loader, device):
    """Return (labels:int ndarray, probs:float ndarray) for a loader (deterministic)."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(out.view(-1))
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(batch.y.detach().cpu().numpy())
    probs = np.concatenate(all_probs) if len(all_probs) else np.array([], dtype=float)
    labels = np.concatenate(all_labels).astype(int) if len(all_labels) else np.array([], dtype=int)
    return labels, probs


def _mc_dropout_probs_and_labels(model, loader, device, T: int = 5):
    """MC Dropout inference: sample dropout masks T times and return mean/std of probabilities.

    Returns:
        labels: [N] int ndarray
        prob_mean: [N] float ndarray
        prob_std: [N] float ndarray
    """
    # labels are deterministic; read once
    all_labels = []
    for batch in loader:
        all_labels.append(batch.y.detach().cpu().numpy())
    labels = np.concatenate(all_labels).astype(int) if len(all_labels) else np.array([], dtype=int)

    probs_T = []
    model.train()  # IMPORTANT: activates dropout at inference

    with torch.no_grad():
        for _ in range(int(T)):
            all_probs = []
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out.view(-1))
                all_probs.append(probs.detach().cpu().numpy())
            probs_T.append(np.concatenate(all_probs) if len(all_probs) else np.array([], dtype=float))

    probs_T = np.stack(probs_T, axis=0) if len(probs_T) else np.zeros((0, 0), dtype=float)  # [T, N]
    prob_mean = probs_T.mean(axis=0) if probs_T.size else np.array([], dtype=float)
    prob_std = (probs_T.std(axis=0, ddof=1) if int(T) > 1 else np.zeros_like(prob_mean)) if probs_T.size else np.array([], dtype=float)

    model.eval()  # restore
    return labels, prob_mean, prob_std


def _uncertainty_gated_preds(
    y_prob_mean: np.ndarray,
    y_prob_std: np.ndarray,
    threshold: float,
    rel_std_max: float = 0.6,
    eps: float = 1e-12,
) -> np.ndarray:
    """Uncertainty-gated hard predictions.

    Policy:
      - If mean < threshold AND (std / denom) <= rel_std_max  -> predict 0
      - Otherwise (including "uncertain" where std/denom > rel_std_max) -> predict 1

    Where denom = max(mean, eps).

    Rationale:
      The decision threshold is derived from the minimum probability among true-1
      graphs on the TRAIN set; it can be extremely small (e.g. 0.0011 or even
      smaller). The uncertainty test should therefore be relative to the model's
      predicted mean probability itself, not to the (possibly tiny) threshold.

      "Uncertain" cases (std/denom > rel_std_max) are assigned to class 1.
    """
    y_prob_mean = np.asarray(y_prob_mean, dtype=float)
    y_prob_std = np.asarray(y_prob_std, dtype=float)

    # Relative uncertainty: std / max(mean, eps)
    denom = np.maximum(y_prob_mean, eps)
    rel = y_prob_std / denom

    # Uncertain (rel > rel_std_max) → predict 1 by construction.
    return np.where((y_prob_mean < threshold) & (rel <= rel_std_max), 0, 1).astype(int)


# =====================================================
# Plotting utilities (each returns a figure)
# =====================================================

def plot_true_labels_scatter(y_true, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(12, 2.5))
    x = np.arange(len(y_true))
    colors = ["red" if y == 0 else "blue" for y in y_true]

    plt.scatter(x, [0]*len(x), c=colors, alpha=0.6, s=8)
    plt.yticks([])
    plt.xlabel("Graph index")
    plt.title("True Labels")
    save_fig(fig, f"{prefix}_true_labels_scatter.png", plot_dir)
    return fig


def plot_probabilities(y_true, y_prob, threshold, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(12, 4))
    x = np.arange(len(y_true))
    colors = ["red" if y == 0 else "blue" for y in y_true]

    plt.scatter(x, y_prob, c=colors, alpha=0.6, s=10)
    plt.axhline(threshold, color="black", linestyle="--", label=f"threshold={threshold}")
    plt.xlabel("Graph index")
    plt.ylabel("Predicted probability")
    plt.ylim([-0.05, 1.05])
    plt.title("Predicted Probabilities")
    plt.legend()
    save_fig(fig, f"{prefix}_probabilities.png", plot_dir)
    return fig


def plot_misclassifications(y_true, y_pred, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(12, 3))
    x = np.arange(len(y_true))
    mis = (y_true != y_pred)

    plt.scatter(x[mis], y_true[mis], c="orange", s=14, label="Misclassified")
    plt.yticks([0, 1])
    plt.xlabel("Graph index")
    plt.ylabel("True label")
    plt.title("Misclassified Samples")
    plt.legend()
    save_fig(fig, f"{prefix}_misclassifications.png", plot_dir)
    return fig


def plot_pr_curve(y_true, y_prob, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    save_fig(fig, f"{prefix}_pr_curve.png", plot_dir)
    return fig


def plot_roc_curve(y_true, y_prob, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--', color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    save_fig(fig, f"{prefix}_roc_curve.png", plot_dir)
    return fig


def plot_confusion(y_true, y_pred, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(4, 3))
    cm = confusion_matrix(y_true, y_pred)

    if sns is None:
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, int(v), ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
    else:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )

    plt.title("Confusion Matrix")
    save_fig(fig, f"{prefix}_confusion_matrix.png", plot_dir)
    return fig


def plot_prob_hist_per_class(y_true, y_prob, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(6, 4))
    probs_0 = y_prob[y_true == 0]
    probs_1 = y_prob[y_true == 1]
    plt.hist(probs_0, bins=20, alpha=0.5, label="class 0", color="red")
    plt.hist(probs_1, bins=20, alpha=0.5, label="class 1", color="blue")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution per Class")
    plt.legend()
    save_fig(fig, f"{prefix}_prob_hist_per_class.png", plot_dir)
    return fig


def plot_sorted_positive_probs(y_true, y_prob, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(6, 4))
    pos_probs = np.sort(y_prob[y_true == 1])[::-1]
    x = np.arange(len(pos_probs))
    plt.plot(x, pos_probs)
    plt.xlabel("Positive graphs (sorted by p)")
    plt.ylabel("Predicted probability")
    plt.title("Sorted Probabilities for True Positive Class")
    plt.ylim([-0.05, 1.05])
    save_fig(fig, f"{prefix}_sorted_positive_probs.png", plot_dir)
    return fig


# =====================================================
# False Negative Analysis
# =====================================================

def false_negative_analysis(y_true, y_pred, y_prob, dataset, file_ext, plot_dir: Path, prefix: str, make_plots: bool = True, y_prob_std: np.ndarray | None = None):
    """
    Focus on false negatives (true=1, pred=0).
    Always saves CSVs with metadata.

    If make_plots is True, also generates and saves FN-focused plots.

    If y_prob_std is provided, it is saved alongside probability as 'y_prob_std'.
    """
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fn_probs = y_prob[fn_idx]
    fn_stds = None
    if y_prob_std is not None:
        fn_stds = np.asarray(y_prob_std)[fn_idx]

    tp_pos_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    tp_probs = y_prob[tp_pos_idx]
    tp_stds = None
    if y_prob_std is not None:
        tp_stds = np.asarray(y_prob_std)[tp_pos_idx]

    # Build FN metadata
    rows_fn = []
    if fn_stds is None:
        for idx, prob in zip(fn_idx, fn_probs):
            data = dataset[idx]
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.shape[0]
            # assuming undirected with duplicated edges; if directed, remove //2
            num_edges = data.edge_index.shape[1] // 2
            rows_fn.append({
                "index": int(idx),
                "y_true": int(y_true[idx]),
                "y_pred": int(y_pred[idx]),
                "probability": float(prob),
                "num_nodes": int(num_nodes),
                "num_edges": int(num_edges),
            })
    else:
        for idx, prob, std in zip(fn_idx, fn_probs, fn_stds):
            data = dataset[idx]
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.shape[0]
            num_edges = data.edge_index.shape[1] // 2
            rows_fn.append({
                "index": int(idx),
                "y_true": int(y_true[idx]),
                "y_pred": int(y_pred[idx]),
                "probability": float(prob),
                "y_prob_std": float(std),
                "num_nodes": int(num_nodes),
                "num_edges": int(num_edges),
            })

    fn_df = pd.DataFrame(rows_fn)
    fn_csv_path = plot_dir / f"{prefix}_false_negatives.csv"
    fn_df.to_csv(fn_csv_path, index=False)

    # TP positives metadata
    rows_tp = []
    if tp_stds is None:
        for idx, prob in zip(tp_pos_idx, tp_probs):
            data = dataset[idx]
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.shape[0]
            num_edges = data.edge_index.shape[1] // 2
            rows_tp.append({
                "index": int(idx),
                "y_true": int(y_true[idx]),
                "y_pred": int(y_pred[idx]),
                "probability": float(prob),
                "num_nodes": int(num_nodes),
                "num_edges": int(num_edges),
            })
    else:
        for idx, prob, std in zip(tp_pos_idx, tp_probs, tp_stds):
            data = dataset[idx]
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.shape[0]
            num_edges = data.edge_index.shape[1] // 2
            rows_tp.append({
                "index": int(idx),
                "y_true": int(y_true[idx]),
                "y_pred": int(y_pred[idx]),
                "probability": float(prob),
                "y_prob_std": float(std),
                "num_nodes": int(num_nodes),
                "num_edges": int(num_edges),
            })

    tp_df = pd.DataFrame(rows_tp)
    tp_csv_path = plot_dir / f"{prefix}_true_positives_positive_class.csv"
    tp_df.to_csv(tp_csv_path, index=False)

    figs = []
    if not make_plots:
        return figs, fn_idx, fn_probs

    # 1. FN scatter
    fig = plt.figure(figsize=(12, 3))
    plt.scatter(fn_idx, fn_probs, c="orange", s=16)
    plt.xlabel("Graph index")
    plt.ylabel("Predicted probability")
    plt.title("False Negative Predictions (true=1, pred=0)")
    save_fig(fig, f"{prefix}_false_negative_scatter.png", plot_dir)
    figs.append(fig)

    # 2. FN histogram
    fig = plt.figure(figsize=(6, 4))
    if len(fn_probs) > 0:
        plt.hist(fn_probs, bins=20, color="orange", alpha=0.7)
    plt.xlabel("Predicted probability")
    plt.title("False Negative Probability Distribution")
    save_fig(fig, f"{prefix}_false_negative_hist.png", plot_dir)
    figs.append(fig)

    # 3. FN vs TP positive comparison
    fig = plt.figure(figsize=(7, 4))
    if len(tp_probs) > 0:
        plt.hist(tp_probs, bins=20, alpha=0.5, label="TP (positives)", color="blue")
    if len(fn_probs) > 0:
        plt.hist(fn_probs, bins=20, alpha=0.7, label="False Negatives", color="orange")
    plt.xlabel("Predicted probability")
    plt.title("FN vs TP+ Probability Distribution")
    plt.legend()
    save_fig(fig, f"{prefix}_false_negative_vs_tp_hist.png", plot_dir)
    figs.append(fig)

    return figs, fn_idx, fn_probs


# =====================================================
# Metrics row for CSV & PDF
# =====================================================

def compute_metric_row(y_true, y_prob, y_pred, metrics_dict, train_threshold=None, eval_threshold=None):
    """Build metric dictionary for CSV export and PDF summary."""
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_prob = np.array(y_prob).astype(float)

    # Recall on 1
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    P = max((y_true == 1).sum(), 1)
    recall_1 = tp / P

    # Recall on 0
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    N0 = max((y_true == 0).sum(), 1)
    recall_0 = tn / N0

    accuracy = (y_pred == y_true).mean()

    precision = metrics_dict.get("precision", None)
    f1 = metrics_dict.get("f1", None)
    roc_auc = metrics_dict.get("roc_auc", None)
    pr_auc = metrics_dict.get("pr_auc", None)
    neg_removal_fraction = metrics_dict.get("neg_removal_fraction", None)

    # Lowest predicted probability for true-1 graph
    if (y_true == 1).sum() > 0:
        lowest_prob_true1 = float(y_prob[y_true == 1].min())
    else:
        lowest_prob_true1 = None

    # Threshold-based evaluation (use minimum prob among true positives on THIS evaluated set)
    # Useful to quantify how conservative a threshold must be to guarantee zero false negatives.
    train_threshold_eval = None
    if train_threshold is not None:
        train_threshold_eval = evaluate_threshold_from_train(
            float(train_threshold),
            torch.as_tensor(y_true, dtype=torch.long),
            torch.as_tensor(y_prob, dtype=torch.float32),
        )

    eval_threshold_eval = None
    if eval_threshold is not None:
        eval_threshold_eval = evaluate_threshold_from_train(
            float(eval_threshold),
            torch.as_tensor(y_true, dtype=torch.long),
            torch.as_tensor(y_prob, dtype=torch.float32),
        )

    return {
        "accuracy": float(accuracy),
        "recall_1": float(recall_1),
        "recall_0": float(recall_0),
        "precision": float(precision) if precision is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "neg_removal_fraction": float(neg_removal_fraction) if neg_removal_fraction is not None else None,
        "lowest_prob_true1": lowest_prob_true1,
        # Flatten key threshold-eval fields for CSV convenience
        "train_nf_threshold": (train_threshold_eval.get("threshold") if train_threshold_eval else None),
        "train_nf_no_false_negatives": (train_threshold_eval.get("no_false_negatives") if train_threshold_eval else None),
        "train_nf_negatives_below_threshold": (train_threshold_eval.get("negatives_below_threshold") if train_threshold_eval else None),
        "train_nf_pct_negatives_below_threshold": (train_threshold_eval.get("pct_negatives_below_threshold") if train_threshold_eval else None),

        "eval_nf_threshold": (eval_threshold_eval.get("threshold") if eval_threshold_eval else None),
        "eval_nf_no_false_negatives": (eval_threshold_eval.get("no_false_negatives") if eval_threshold_eval else None),
        "eval_nf_negatives_below_threshold": (eval_threshold_eval.get("negatives_below_threshold") if eval_threshold_eval else None),
        "eval_nf_pct_negatives_below_threshold": (eval_threshold_eval.get("pct_negatives_below_threshold") if eval_threshold_eval else None),
    }


def metrics_figure(metric_row):
    """
    Create a simple figure summarizing metrics as text,
    to be inserted as first page in the PDF.
    """
    fig = plt.figure(figsize=(8, 6))
    plt.axis("off")

    lines = []
    for k, v in metric_row.items():
        lines.append(f"{k}: {v}")
    text = "\n".join(lines)

    plt.text(0.05, 0.95, text, va="top", fontsize=12)
    return fig


# =====================================================
# CLI parsing
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run saved GNN model on data")

    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Checkpoint file name inside model_dir (e.g. best_model.pt)"
    )

    parser.add_argument(
        "--data_file", type=str, required=True,
        help="Dataset graph file (.csv or .npz) to evaluate (e.g. den_graph_data_9.npz)"
    )

    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Config YAML"
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--embedding", action="store_true", help="Save per-graph embeddings (GIN pre-pooling jump output) to .npy")

    parser.add_argument(
        "--dropout",
        action="store_true",
        help="Enable MC Dropout at inference: run T stochastic passes and output mean/std of probabilities."
    )
    parser.add_argument(
        "--T",
        type=int,
        default=5,
        help="Number of MC Dropout samples (default: 5). Only used with --dropout."
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots and a PDF report. If not set, no plots/PDF are generated."
    )

    return parser.parse_args()


# =====================================================
# Dataset loading
# =====================================================

def extract_file_ext(path: Path):
    """
    Extract file_ext from filename.

    den_graph_data_12_full.npz -> "12_full"
    den_graph_data_12.npz      -> "12"
    tomato.npz                 -> "tomato"
    """
    stem = path.stem  # e.g. "den_graph_data_12_full"

    prefix = "den_graph_data_"
    if stem.startswith(prefix):
        return stem[len(prefix):]  # strip prefix, keep "12_full"

    return stem  # fallback


def load_dataset(data_file, selected_features, scaler=None):
    """
    Use only file_ext = exact suffix extracted from filename.
    """
    p = Path(data_file)

    if p.suffix not in {".csv", ".npz"}:
        raise ValueError("data_file must end in .csv or .npz")

    # extract exact extension from filename
    file_ext = extract_file_ext(p)
    print(f"[INFO] Using file_ext: {file_ext}")

    # Auto-detect best parallelism settings
    n_jobs, chunk_size = autotune_resources()
    print(f"[INFO] autotune_resources → n_jobs={n_jobs}, chunk_size={chunk_size}")

    # Call dataset builder with tuned values
    ds, scaler, feats = create_simple_dataset(
        file_ext=file_ext,
        selected_features=selected_features,
        normalize=True,
        data_dir=str(p.parent),      # directory containing the NPZ
        scaler=scaler,
        n_jobs=n_jobs,
        chunk_size=chunk_size
    )

    return ds, scaler, file_ext


def _train_fileexts_from_config(data_cfg):
    """Normalize train_loop_order from config into a list of file_ext strings."""
    train_order = data_cfg.get("train_loop_order", [])
    if train_order is None:
        return []
    if isinstance(train_order, (str, int)):
        return [str(train_order)]
    return [str(x) for x in train_order]


def _data_dir_from_config(data_cfg, config_path: Path):
    """Resolve base_dir robustly.

    Priority:
      1) Absolute base_dir as-is.
      2) Relative to repo root (parent of this GNN dir).
      3) Relative to current working directory.
      4) Fallback: relative to config file (legacy).
    """
    base_dir = data_cfg.get("base_dir", "")
    if not base_dir:
        return None

    p = Path(str(base_dir))
    if p.is_absolute():
        return p

    # repo root = .../ML-correlator
    repo_root = Path(__file__).resolve().parents[1]
    cand = (repo_root / p).resolve()
    if cand.exists():
        return cand

    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return cand

    # Legacy fallback: relative to the config file location
    return (config_path.parent / p).resolve()


def normalize_loop_order(value):
    """Normalize loop order input into a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        return [value.strip()]
    return [str(value)]


# =====================================================
# Main evaluation
# =====================================================

def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    features_cfg = cfg["features"]
    experiment_cfg = cfg["experiment"]
    selected_features = features_cfg["selected_features"]

    wandb_project = experiment_cfg.get("wandb_project", "default_project")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    print(f"Loading dataset from: {args.data_file}")
    dataset, _, file_ext = load_dataset(
        data_file=args.data_file,
        selected_features=selected_features
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    num_features = dataset[0].x.shape[1]
    N = len(dataset)

    # Recreate model from config
    model = create_gnn_model(
        architecture=model_cfg["name"],
        num_features=num_features,
        hidden_dim=model_cfg["hidden_channels"],
        num_classes=1,
        dropout=model_cfg.get("dropout", 0.0),
        num_layers=model_cfg["num_layers"],
    ).to(device)

    # Output directory based on wandb_project, model_name and file_ext
    model_stem = Path(args.model_name).stem
    prefix = f"{model_stem}_{file_ext}"
    output_dir = Path("models") / wandb_project / prefix
    ensure_dir(output_dir)
    print(f"[INFO] Saving outputs to: {output_dir}")

    # Load checkpoint
    checkpoint_path = Path(experiment_cfg["model_dir"]) / args.model_name
    print(f"Loading model checkpoint: {checkpoint_path}")

    checkpoint_raw = torch.load(checkpoint_path, map_location=device)

    def _extract_state_dict(ckpt):
        """Return a plausible model state_dict from a checkpoint object."""
        if isinstance(ckpt, dict):
            # common wrappers
            for k in ("model_state_dict", "state_dict", "model", "net"):
                v = ckpt.get(k)
                if isinstance(v, dict):
                    return v
            # already a state_dict
            if all(isinstance(k, str) for k in ckpt.keys()):
                return ckpt
        return ckpt

    state_dict = _extract_state_dict(checkpoint_raw)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Provide a more actionable error message.
        sd_keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
        head = sd_keys[:20]
        raise RuntimeError(
            "Failed to load checkpoint into the reconstructed model. "
            "This usually means the evaluation config's model architecture/hidden_channels/num_layers/dropout "
            "does not match the checkpoint (e.g., loading a GAT checkpoint with a GIN config), "
            "or the checkpoint is wrapped under a different key.\n\n"
            f"Checkpoint type: {type(checkpoint_raw)} | extracted_state_dict_keys(sample): {head}\n"
            f"Config model: name={model_cfg.get('name')} hidden_channels={model_cfg.get('hidden_channels')} num_layers={model_cfg.get('num_layers')} dropout={model_cfg.get('dropout')}\n\n"
            f"Original load_state_dict error:\n{e}"
        )

    model.eval()

    # -----------------------------------------------------
    # Compute train_min_prob by running checkpoint over ALL configured training loops
    # -----------------------------------------------------
    train_threshold_eval = None
    train_min_prob = None

    data_cfg = cfg.get("data", {})
    train_fileexts = _train_fileexts_from_config(data_cfg)
    train_base_dir = _data_dir_from_config(data_cfg, Path(args.config).resolve())

    # IMPORTANT: mimic training-time preprocessing:
    #   - fit a scaler on the first training loop, then reuse it
    #   - keep a running max feature width for padding
    # This is required for train_min_prob to match wandb's train_lowest_prob_true1.
    train_labels_all, train_probs_all = [], []
    train_scaler = None
    train_max_features = None

    if train_base_dir is None or len(train_fileexts) == 0:
        print("[WARN] Could not resolve training datasets from config; cannot compute train_min_prob.")
    else:
        print("\nComputing TRAIN threshold (min prob among true positives on TRAIN set)…")
        print(f"[INFO] train_loop_order={train_fileexts}")
        print(f"[INFO] base_dir={train_base_dir}")

        # 1) Load all training datasets with training-like scaler/max_features reuse
        train_datasets = []
        for tr_ext in train_fileexts:
            train_path = train_base_dir / f"den_graph_data_{tr_ext}.npz"
            if not train_path.exists():
                train_path = train_base_dir / f"den_graph_data_{tr_ext}.csv"

            if not train_path.exists():
                print(f"[WARN] Training data file not found for loop {tr_ext}: expected {train_base_dir}/den_graph_data_{tr_ext}.npz (.csv)")
                continue

            # Use create_simple_dataset directly so we can control scaler/max_features reuse.
            # (load_dataset() always re-autotunes and doesn't let us keep max_features.)
            n_jobs, chunk_size = autotune_resources()
            tr_ds, train_scaler, feats = create_simple_dataset(
                file_ext=extract_file_ext(Path(train_path)),
                selected_features=selected_features,
                normalize=True,
                data_dir=str(Path(train_path).parent),
                scaler=train_scaler,
                max_features=train_max_features,
                n_jobs=n_jobs,
                chunk_size=chunk_size,
            )

            # Update running max feature width after reading this dataset
            if train_max_features is None or int(feats) > int(train_max_features):
                train_max_features = int(feats)

            train_datasets.append(tr_ds)

        if len(train_datasets) == 0:
            print("[WARN] No training datasets were loaded; cannot compute train_min_prob.")
        else:
            # 2) Pad all training datasets to a consistent width (training does this)
            final_max_features = max(int(d.x.shape[1]) for ds in train_datasets for d in ds)
            for ds in train_datasets:
                for data in ds:
                    n_nodes, n_feats = data.x.shape
                    if n_feats < final_max_features:
                        pad = torch.zeros(n_nodes, final_max_features - n_feats, dtype=data.x.dtype)
                        data.x = torch.cat([data.x, pad], dim=1)

            # 3) Run the checkpoint over each training dataset and aggregate labels/probs
            for tr_ds in train_datasets:
                tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False)
                tr_labels, tr_probs = _predict_probs_and_labels(model, tr_loader, device)
                train_labels_all.append(tr_labels)
                train_probs_all.append(tr_probs)

            train_labels_all = np.concatenate(train_labels_all).astype(int) if len(train_labels_all) else np.array([], dtype=int)
            train_probs_all = np.concatenate(train_probs_all).astype(float) if len(train_probs_all) else np.array([], dtype=float)

            train_min_prob = compute_min_positive_prob(
                torch.as_tensor(train_labels_all, dtype=torch.long),
                torch.as_tensor(train_probs_all, dtype=torch.float32),
            )

            if train_min_prob is None:
                print("[WARN] TRAIN set contains no positives; cannot compute train_min_prob.")
            else:
                print(f"[INFO] train_min_prob (TRAIN min positive prob) = {float(train_min_prob):.10f}")

    if train_min_prob is None:
        raise RuntimeError(
            "Could not compute training-derived threshold (train_min_prob is None). "
            "Check data.base_dir and data.train_loop_order in the config, and ensure training files exist and contain positives."
        )

    # Use TRAIN-derived threshold everywhere (plots/metrics/predictions).
    train_threshold_used = float(train_min_prob)
    threshold_used = train_threshold_used  # backwards-compat alias; do not overwrite with eval_min_prob
    print(f"[INFO] train_threshold_used (TRAIN min-positive) = {train_threshold_used:.18e}")

    # =====================================================
    # Evaluate (standard metrics)
    # =====================================================
    # If --dropout is enabled, we report metrics based on the uncertainty-gated predictions.
    # Otherwise we keep the original evaluate(...) behavior.

    print("\n=== MODEL EVALUATION (standard metrics) ===")

    if args.dropout:
        # We'll compute probabilities first and then compute gated metrics.
        # Loss is not computed in this path.
        avg_loss = float("nan")
        accuracy = float("nan")
        metrics = {"roc_auc": float("nan"), "pr_auc": float("nan")}
    else:
        avg_loss, accuracy, metrics = evaluate(
            model, loader, device=device,
            threshold=train_threshold_used,
            log_threshold_curves=True,
            split_name="eval",
        )

    # Recompute probabilities for downstream analysis/plots
    if args.dropout:
        print(f"[INFO] MC Dropout enabled for outputs (T={int(args.T)})")
        all_labels, all_probs_mean, all_probs_std = _mc_dropout_probs_and_labels(model, loader, device, T=int(args.T))
        y_prob_used = all_probs_mean
    else:
        all_labels, all_probs_det = _predict_probs_and_labels(model, loader, device)
        all_probs_std = None
        y_prob_used = all_probs_det

    # NOTE: all downstream metrics/plots/FN analysis use y_prob_used (MC mean if --dropout).
    if args.dropout:
        all_preds = _uncertainty_gated_preds(
            y_prob_mean=y_prob_used,
            y_prob_std=all_probs_std,
            threshold=train_threshold_used,
            rel_std_max=0.6,
        )
        gated_metrics = _binary_metrics_from_probs(all_labels, y_prob_used, all_preds)
        # standard-metrics printout uses gated values
        print(f"\nLoss:      {avg_loss}")
        print(f"Accuracy:  {gated_metrics['accuracy']:.4f}" if gated_metrics["accuracy"] is not None else "Accuracy:  None")
        if gated_metrics.get("roc_auc") is not None:
            print(f"ROC-AUC:   {gated_metrics['roc_auc']:.4f}")
        else:
            print("ROC-AUC:   None")
        if gated_metrics.get("pr_auc") is not None:
            print(f"PR-AUC:    {gated_metrics['pr_auc']:.4f}")
        else:
            print("PR-AUC:    None")
        # also pass these through for metric_row
        metrics.update({
            "roc_auc": gated_metrics.get("roc_auc"),
            "pr_auc": gated_metrics.get("pr_auc"),
            "precision": gated_metrics.get("precision"),
            "f1": gated_metrics.get("f1"),
        })
    else:
        all_preds = (y_prob_used >= train_threshold_used).astype(int)
        print(f"\nLoss:      {avg_loss:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        if metrics.get("neg_removal_fraction") is not None:
            print(f"Neg-removal fraction (true 0s below min true-1 prob): {metrics['neg_removal_fraction']:.4f}")

    # Optionally compute eval_min_prob on the evaluated dataset itself (diagnostic only; never used as main threshold)
    eval_min_prob = compute_min_positive_prob(
        torch.as_tensor(all_labels, dtype=torch.long),
        torch.as_tensor(y_prob_used, dtype=torch.float32),
    )
    if eval_min_prob is not None:
        print(f"[INFO] eval_min_prob (EVAL min-positive; diagnostic only) = {float(eval_min_prob):.18e}")
        print(f"[INFO] Sanity: threshold used for preds/plots remains train_threshold_used = {train_threshold_used:.18e}")

    # -----------------------------------------------------
    # Threshold-based evaluations (two distinct meanings)
    # -----------------------------------------------------

    print("\n=== THRESHOLD EVALUATION (A) : TRAIN-derived threshold (train_min_prob) applied to THIS dataset ===")
    print(f"[INFO] threshold_source=TRAIN (min prob among y=1 on TRAIN) → threshold={train_threshold_used:.10f}")

    if args.dropout:
        gated_fn = int(((all_labels == 1) & (all_preds == 0)).sum())
        gated_tp = int(((all_labels == 1) & (all_preds == 1)).sum())
        print("\n============================================================")
        print(f"GATED (mean+std) evaluation @ TRAIN threshold={train_threshold_used:.4f} (rel_std_max=0.6)")
        print("============================================================")
        print(f"False Negatives: {gated_fn} | True Positives: {gated_tp}")
        print("============================================================")

        print("\n[INFO] MEAN-ONLY reference below (ignores std/mean gate)")

    # Mean-only evaluation (printed exactly once)
    train_threshold_eval = evaluate_threshold_from_train(
        float(train_threshold_used),
        torch.as_tensor(all_labels, dtype=torch.long),
        torch.as_tensor(y_prob_used, dtype=torch.float32),
    )

    if eval_min_prob is not None:
        print("\n=== THRESHOLD EVALUATION (B) : EVAL-derived threshold (eval_min_prob) on THIS dataset (diagnostic) ===")
        print(f"[INFO] threshold_source=EVAL (min prob among y=1 on THIS dataset) → threshold={float(eval_min_prob):.10f}")

        if args.dropout:
            eval_preds_gated = _uncertainty_gated_preds(
                y_prob_mean=y_prob_used,
                y_prob_std=all_probs_std,
                threshold=float(eval_min_prob),
                rel_std_max=0.6,
            )
            gated_fn = int(((all_labels == 1) & (eval_preds_gated == 0)).sum())
            gated_tp = int(((all_labels == 1) & (eval_preds_gated == 1)).sum())
            print("\n============================================================")
            print(f"GATED (mean+std) evaluation @ EVAL threshold={float(eval_min_prob):.4f} (rel_std_max=0.6)")
            print("============================================================")
            print(f"False Negatives: {gated_fn} | True Positives: {gated_tp}")
            print("============================================================")

            print("\n[INFO] MEAN-ONLY reference below (ignores std/mean gate)")

        # Mean-only evaluation (printed exactly once)
        _ = evaluate_threshold_from_train(
            float(eval_min_prob),
            torch.as_tensor(all_labels, dtype=torch.long),
            torch.as_tensor(y_prob_used, dtype=torch.float32),
        )

    # =====================================================
    # Physics-oriented scalar metrics
    # =====================================================
    print(f"\n=== PHYSICS-ORIENTED METRICS (threshold = {train_threshold_used:.6f}) ===")

    P = (all_labels == 1).sum()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    ansatz_size = (all_preds == 1).sum()
    ansatz_fraction = ansatz_size / N if N > 0 else 0.0
    reduction_factor = N / ansatz_size if ansatz_size > 0 else np.inf
    recall_pos = tp / P if P > 0 else 0.0

    print(f"Total graphs (N):       {N}")
    print(f"True positives (P):     {P}")
    print(f"Ansatz size:            {ansatz_size}")
    print(f"Ansatz fraction:        {ansatz_fraction:.4f}")
    print(f"Reduction factor:       {reduction_factor:.2f}x")
    print(f"Recall on contributing graphs (coverage): {recall_pos:.4f}")
    print(f"False negatives (missed contributors):    {fn}")

    # =====================================================
    # Metrics CSV for this dataset
    # =====================================================
    metric_row = compute_metric_row(
        y_true=all_labels,
        y_prob=y_prob_used,
        y_pred=all_preds,
        metrics_dict=metrics,
        train_threshold=train_threshold_used,
        eval_threshold=(float(eval_min_prob) if eval_min_prob is not None else None),
    )

    # =====================================================
    # Always write FN/TP metadata CSVs
    # =====================================================
    fn_figs, fn_idx, fn_probs = false_negative_analysis(
        all_labels, all_preds, y_prob_used, dataset, file_ext, output_dir, prefix,
        make_plots=bool(args.plots),
        y_prob_std=(all_probs_std if args.dropout else None)
    )

    # =====================================================
    # Generate plots (only if --plots)
    # =====================================================
    if args.plots:
        print("\n=== Generating Plots and PDF Report ===")

        figs = []
        figs.append(plot_true_labels_scatter(all_labels, output_dir, prefix))
        figs.append(plot_probabilities(all_labels, y_prob_used, threshold=train_threshold_used,
                                       plot_dir=output_dir, prefix=prefix))
        figs.append(plot_misclassifications(all_labels, all_preds, output_dir, prefix))
        figs.append(plot_pr_curve(all_labels, y_prob_used, output_dir, prefix))
        figs.append(plot_roc_curve(all_labels, y_prob_used, output_dir, prefix))
        figs.append(plot_confusion(all_labels, all_preds, output_dir, prefix))
        figs.append(plot_prob_hist_per_class(all_labels, y_prob_used, output_dir, prefix))
        figs.append(plot_sorted_positive_probs(all_labels, y_prob_used, output_dir, prefix))

        # Add FN-focused figures
        figs.extend(fn_figs)

        # =====================================================
        # Save PDF report (metrics page + all figs)
        # =====================================================
        pdf_path = output_dir / f"{prefix}_evaluation_report.pdf"
        with PdfPages(pdf_path) as pdf:
            # First page: metrics summary
            metrics_fig = metrics_figure(metric_row)
            pdf.savefig(metrics_fig, bbox_inches="tight")
            plt.close(metrics_fig)

            # Then all plot figures
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved PDF report → {pdf_path}")
    else:
        print("\n[INFO] --plots not set; skipping plot/PDF generation.")

    # =====================================================
    # Save predictions CSV (always)
    # =====================================================
    pred_csv_path = output_dir / f"{prefix}_predictions.csv"

    df_dict = {
        "y_true": all_labels,
        "y_pred": all_preds,
    }

    # If dropout is enabled, expose mean/std explicitly as additional columns.
    # NOTE: when --dropout is on, we intentionally do NOT write a plain 'y_prob' column.
    if args.dropout:
        df_dict["y_prob_mean"] = y_prob_used
        df_dict["y_prob_std"] = all_probs_std
    else:
        df_dict["y_prob"] = y_prob_used

    df = pd.DataFrame(df_dict)
    df.to_csv(pred_csv_path, index=False)

    print(f"Saved predictions CSV → {pred_csv_path}")


if __name__ == "__main__":
    main()