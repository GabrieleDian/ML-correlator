#!/usr/bin/env python3
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader

from graph_builder import create_simple_dataset
from GNN_architectures import create_gnn_model
from training_utils import evaluate
from load_features import autotune_resources

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix
)

import os

# =====================================================
# Utilities
# =====================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, filename: str, plot_dir: Path):
    ensure_dir(plot_dir)
    fig.savefig(plot_dir / filename, dpi=200, bbox_inches="tight")


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

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
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

def false_negative_analysis(y_true, y_pred, y_prob, dataset, file_ext, plot_dir: Path, prefix: str):
    """
    Focus on false negatives (true=1, pred=0).
    Save CSV with metadata and generate FN-focused plots.
    """
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fn_probs = y_prob[fn_idx]

    tp_pos_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    tp_probs = y_prob[tp_pos_idx]

    # Build FN metadata
    rows_fn = []
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

    fn_df = pd.DataFrame(rows_fn)
    fn_csv_path = plot_dir / f"{prefix}_false_negatives.csv"
    fn_df.to_csv(fn_csv_path, index=False)

    # TP positives metadata (optional)
    rows_tp = []
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

    tp_df = pd.DataFrame(rows_tp)
    tp_csv_path = plot_dir / f"{prefix}_true_positives_positive_class.csv"
    tp_df.to_csv(tp_csv_path, index=False)

    figs = []

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
# Threshold sweep for recall vs ansatz size
# =====================================================

def threshold_sweep_metrics(y_true, y_prob, num_thresholds=50):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    recalls = []
    fn_counts = []
    ansatz_fracs = []

    N = len(y_true)
    P = (y_true == 1).sum()

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tp = ((y_pred_t == 1) & (y_true == 1)).sum()
        fn = ((y_pred_t == 0) & (y_true == 1)).sum()
        ansatz_size = (y_pred_t == 1).sum()
        recall = tp / P if P > 0 else 0.0
        ansatz_frac = ansatz_size / N if N > 0 else 0.0

        recalls.append(recall)
        fn_counts.append(fn)
        ansatz_fracs.append(ansatz_frac)

    return thresholds, np.array(recalls), np.array(fn_counts), np.array(ansatz_fracs)


def plot_recall_vs_ansatz(ansatz_fracs, recalls, plot_dir: Path, prefix: str):
    fig = plt.figure(figsize=(6, 5))
    plt.plot(ansatz_fracs, recalls, marker="o")
    plt.xlabel("Ansatz fraction (fraction of graphs kept)")
    plt.ylabel("Recall (physics coverage)")
    plt.title("Recall vs Ansatz Size")
    plt.grid(True)
    save_fig(fig, f"{prefix}_recall_vs_ansatz.png", plot_dir)
    return fig


# =====================================================
# Metrics row for CSV & PDF
# =====================================================

def compute_metric_row(y_true, y_prob, y_pred, metrics_dict):
    """
    Build metric dictionary for CSV export and PDF summary.
    """
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
    safe_frac = metrics_dict.get("safe_ansatz_fraction", None)

    # Lowest predicted probability for true-1 graph
    if (y_true == 1).sum() > 0:
        lowest_prob_true1 = float(y_prob[y_true == 1].min())
    else:
        lowest_prob_true1 = None

    return {
        "accuracy": float(accuracy),
        "recall_1": float(recall_1),
        "recall_0": float(recall_0),
        "precision": float(precision) if precision is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "safe_ansatz_fraction": float(safe_frac) if safe_frac is not None else None,
        "lowest_prob_true1": lowest_prob_true1,
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
        dropout=model_cfg["dropout"],
        num_layers=model_cfg["num_layers"]
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Evaluate (for standard metrics)
    print("\n=== MODEL EVALUATION (standard metrics) ===")

    avg_loss, accuracy, metrics = evaluate(
        model, loader, device=device,
        threshold=args.threshold,
        log_threshold_curves=True,
        split_name="eval"
    )

    print(f"\nLoss:      {avg_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Safe-Ansatz Fraction: {metrics['safe_ansatz_fraction']:.4f}")
    if "precision" in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
    if "recall" in metrics:
        print(f"Recall:    {metrics['recall']:.4f}")
    if "f1" in metrics:
        print(f"F1 score:  {metrics['f1']:.4f}")

    # Recompute probabilities for physics-oriented analysis
    print("\nRecomputing probabilities for physics-oriented analysis…")

    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(out.view(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels).astype(int)
    all_preds = (all_probs >= args.threshold).astype(int)

    # =====================================================
    # Physics-oriented scalar metrics (reduction & FN)
    # =====================================================
    print("\n=== PHYSICS-ORIENTED METRICS (threshold = {:.3f}) ===".format(args.threshold))

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
    # Threshold sweep: recall vs ansatz
    # =====================================================
    print("\n=== THRESHOLD SWEEP (no automatic choice) ===")
    thresholds, recalls, fn_counts, ansatz_fracs = threshold_sweep_metrics(all_labels, all_probs)

    # Print a small table
    print(f"{'thr':>6} {'recall':>8} {'FN':>6} {'ansatz_frac':>12}")
    for t, r, fn_t, af in zip(thresholds, recalls, fn_counts, ansatz_fracs):
        print(f"{t:6.3f} {r:8.4f} {int(fn_t):6d} {af:12.4f}")

    # =====================================================
    # Metrics CSV for this dataset
    # =====================================================
    metric_row = compute_metric_row(
        y_true=all_labels,
        y_prob=all_probs,
        y_pred=all_preds,
        metrics_dict=metrics
    )
    metric_df = pd.DataFrame([metric_row])
    metric_csv_path = output_dir / f"{prefix}_metrics.csv"
    metric_df.to_csv(metric_csv_path, index=False)
    print(f"[INFO] Saved metrics summary → {metric_csv_path}")

    # =====================================================
    # Generate plots
    # =====================================================
    print("\n=== Generating Plots and PDF Report ===")

    figs = []
    figs.append(plot_true_labels_scatter(all_labels, output_dir, prefix))
    figs.append(plot_probabilities(all_labels, all_probs, threshold=args.threshold,
                                   plot_dir=output_dir, prefix=prefix))
    figs.append(plot_misclassifications(all_labels, all_preds, output_dir, prefix))
    figs.append(plot_pr_curve(all_labels, all_probs, output_dir, prefix))
    figs.append(plot_roc_curve(all_labels, all_probs, output_dir, prefix))
    figs.append(plot_confusion(all_labels, all_preds, output_dir, prefix))
    figs.append(plot_prob_hist_per_class(all_labels, all_probs, output_dir, prefix))
    figs.append(plot_sorted_positive_probs(all_labels, all_probs, output_dir, prefix))

    # Recall vs Ansatz plot
    figs.append(plot_recall_vs_ansatz(ansatz_fracs, recalls, output_dir, prefix))

    # False Negative Analysis with metadata
    print("\n=== False Negative Analysis ===")
    fn_figs, fn_idx, fn_probs = false_negative_analysis(
        all_labels, all_preds, all_probs, dataset, file_ext, output_dir, prefix
    )
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

    # =====================================================
    # Save predictions CSV
    # =====================================================
    pred_csv_path = output_dir / f"{prefix}_predictions.csv"
    df = pd.DataFrame({
        "y_true": all_labels,
        "y_prob": all_probs,
        "y_pred": all_preds
    })
    df.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions → {pred_csv_path}")
    print(f"False negatives with metadata → {output_dir / (prefix + '_false_negatives.csv')}")
    print(f"True positive positives with metadata → {output_dir / (prefix + '_true_positives_positive_class.csv')}")


if __name__ == "__main__":
    main()
