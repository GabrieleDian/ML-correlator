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
import re

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

def plot_true_labels_scatter(y_true, plot_dir):
    fig = plt.figure(figsize=(12, 2.5))
    x = np.arange(len(y_true))
    colors = ["red" if y == 0 else "blue" for y in y_true]

    plt.scatter(x, [0]*len(x), c=colors, alpha=0.6, s=8)
    plt.yticks([])
    plt.xlabel("Graph index")
    plt.title("True Labels")
    save_fig(fig, "true_labels_scatter.png", plot_dir)
    return fig


def plot_probabilities(y_true, y_prob, threshold, plot_dir):
    fig = plt.figure(figsize=(12, 4))
    x = np.arange(len(y_true))
    colors = ["red" if y == 0 else "blue" for y in y_true]

    plt.scatter(x, y_prob, c=colors, alpha=0.6, s=10)
    plt.axhline(threshold, color="black", linestyle="--")
    plt.xlabel("Graph index")
    plt.ylabel("Predicted probability")
    plt.ylim([-0.05, 1.05])
    plt.title("Predicted Probabilities")
    save_fig(fig, "probabilities.png", plot_dir)
    return fig


def plot_misclassifications(y_true, y_pred, plot_dir):
    fig = plt.figure(figsize=(12, 3))
    x = np.arange(len(y_true))
    mis = (y_true != y_pred)

    plt.scatter(x[mis], y_true[mis], c="orange", s=14)
    plt.yticks([0, 1])
    plt.xlabel("Graph index")
    plt.title("Misclassified Samples")
    save_fig(fig, "misclassifications.png", plot_dir)
    return fig


def plot_pr_curve(y_true, y_prob, plot_dir):
    fig = plt.figure(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    save_fig(fig, "pr_curve.png", plot_dir)
    return fig


def plot_roc_curve(y_true, y_prob, plot_dir):
    fig = plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--',color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    save_fig(fig, "roc_curve.png", plot_dir)
    return fig


def plot_confusion(y_true, y_pred, plot_dir):
    fig = plt.figure(figsize=(4, 3))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.title("Confusion Matrix")
    save_fig(fig, "confusion_matrix.png", plot_dir)
    return fig


def plot_prob_hist_per_class(y_true, y_prob, plot_dir):
    fig = plt.figure(figsize=(6, 4))
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.5, color="red", label="Class 0")
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.5, color="blue", label="Class 1")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution per Class")
    plt.legend()
    save_fig(fig, "prob_hist_per_class.png", plot_dir)
    return fig


def plot_sorted_positive_probs(y_true, y_prob, plot_dir):
    fig = plt.figure(figsize=(6,4))
    pos_probs = np.sort(y_prob[y_true == 1])[::-1]
    x = np.arange(len(pos_probs))
    plt.plot(x, pos_probs)
    plt.xlabel("Sorted positive indices")
    plt.ylabel("Predicted probability")
    plt.title("Sorted positive probabilities")
    plt.ylim([-0.05,1.05])
    save_fig(fig, "sorted_positive_probs.png", plot_dir)
    return fig



# =====================================================
# FN analysis
# =====================================================

def false_negative_analysis(y_true, y_pred, y_prob, dataset, file_ext, plot_dir):
    fn_idx = np.where((y_true==1) & (y_pred==0))[0]
    fn_probs = y_prob[fn_idx]

    tp_idx = np.where((y_true==1) & (y_pred==1))[0]
    tp_probs = y_prob[tp_idx]

    rows = []
    for idx, prob in zip(fn_idx, fn_probs):
        data = dataset[idx]
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]//2
        rows.append({
            "index": int(idx),
            "probability": float(prob),
            "num_nodes": num_nodes,
            "num_edges": num_edges
        })

    df = pd.DataFrame(rows)
    df.to_csv(plot_dir/"false_negatives.csv", index=False)

    figs = []

    fig = plt.figure(figsize=(12,3))
    plt.scatter(fn_idx, fn_probs, s=16, c="orange")
    plt.title("False Negatives")
    save_fig(fig, "fn_scatter.png", plot_dir)
    figs.append(fig)

    fig = plt.figure(figsize=(6,4))
    plt.hist(fn_probs, bins=20, color="orange")
    plt.title("False Negative Probability Distribution")
    save_fig(fig, "fn_hist.png", plot_dir)
    figs.append(fig)

    return figs



# =====================================================
# Threshold sweep
# =====================================================

def threshold_sweep_metrics(y_true, y_prob):
    thresholds = np.linspace(0,1,50)
    recalls = []
    ansatz_fracs = []
    fn_counts = []

    N = len(y_true)
    P = (y_true==1).sum()

    for t in thresholds:
        y_pred_t = (y_prob>=t).astype(int)
        tp = ((y_pred_t==1)&(y_true==1)).sum()
        fn = ((y_pred_t==0)&(y_true==1)).sum()
        ansatz_size = (y_pred_t==1).sum()

        recalls.append(tp/P if P>0 else 0)
        fn_counts.append(fn)
        ansatz_fracs.append(ansatz_size/N)

    return thresholds, np.array(recalls), np.array(fn_counts), np.array(ansatz_fracs)



def plot_recall_vs_ansatz(ansatz_fracs, recalls, plot_dir):
    fig = plt.figure(figsize=(6,5))
    plt.plot(ansatz_fracs, recalls, marker="o")
    plt.xlabel("Ansatz fraction")
    plt.ylabel("Recall (coverage)")
    plt.title("Recall vs Ansatz size")
    plt.grid(True)
    save_fig(fig, "recall_vs_ansatz.png", plot_dir)
    return fig



# =====================================================
# CLI parsing
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run saved GNN model on data")
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--data_file", required=True, type=str)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


# =====================================================
# Dataset loading
# =====================================================

def extract_file_ext(path: Path):
    stem = path.stem
    prefix = "den_graph_data_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def load_dataset(data_file, selected_features, scaler=None):
    p = Path(data_file)
    file_ext = extract_file_ext(p)
    print(f"[INFO] file_ext = {file_ext}")

    n_jobs, chunk_size = autotune_resources()
    print(f"[INFO] n_jobs={n_jobs}, chunk_size={chunk_size}")

    ds, scaler, feats = create_simple_dataset(
        file_ext=file_ext,
        selected_features=selected_features,
        normalize=True,
        data_dir=str(p.parent),
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

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    features_cfg = cfg["features"]
    experiment_cfg = cfg["experiment"]
    selected_features = features_cfg["selected_features"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset from {args.data_file}")
    dataset, _, file_ext = load_dataset(args.data_file, selected_features)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    num_features = dataset[0].x.shape[1]

    # Create model
    model = create_gnn_model(
        architecture=model_cfg["name"],
        num_features=num_features,
        hidden_dim=model_cfg["hidden_channels"],
        num_classes=1,
        dropout=model_cfg["dropout"],
        num_layers=model_cfg["num_layers"]
    ).to(device)

    # Prepare output directory
    model_base = Path(args.model_name).stem
    plot_dir = Path("models") / model_base
    ensure_dir(plot_dir)
    print(f"[INFO] Saving plots to {plot_dir}")

    # Load weights
    checkpoint_path = Path(experiment_cfg["model_dir"]) / args.model_name
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Evaluate standard metrics
    avg_loss, accuracy, metrics = evaluate(
        model, loader, device, threshold=args.threshold,
        log_threshold_curves=False
    )

    # Recompute probabilities for analysis
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(out.view(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_labels = np.concatenate(all_labels).astype(int)
    all_probs = np.concatenate(all_probs)
    all_preds = (all_probs >= args.threshold).astype(int)

    # Summary metrics
    N = len(all_labels)
    P = (all_labels==1).sum()
    tp = ((all_preds==1)&(all_labels==1)).sum()
    fn = ((all_preds==0)&(all_labels==1)).sum()
    ansatz_size = (all_preds==1).sum()

    # -----------------------
    # Generate Plots
    # -----------------------
    figs = []
    figs.append(plot_true_labels_scatter(all_labels, plot_dir))
    figs.append(plot_probabilities(all_labels, all_probs, args.threshold, plot_dir))
    figs.append(plot_misclassifications(all_labels, all_preds, plot_dir))
    figs.append(plot_pr_curve(all_labels, all_probs, plot_dir))
    figs.append(plot_roc_curve(all_labels, all_probs, plot_dir))
    figs.append(plot_confusion(all_labels, all_preds, plot_dir))
    figs.append(plot_prob_hist_per_class(all_labels, all_probs, plot_dir))
    figs.append(plot_sorted_positive_probs(all_labels, all_probs, plot_dir))

    # Threshold sweep plot
    thr, recs, fns, ansfr = threshold_sweep_metrics(all_labels, all_probs)
    figs.append(plot_recall_vs_ansatz(ansfr, recs, plot_dir))

    # False negatives
    fn_figs = false_negative_analysis(all_labels, all_preds, all_probs, dataset, file_ext, plot_dir)
    figs.extend(fn_figs)

    # Save full PDF report
    pdf_path = plot_dir / "evaluation_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")

    print(f"[INFO] Saved PDF report → {pdf_path}")

    # Save predictions CSV
    pred_csv = plot_dir / f"predictions_{Path(args.data_file).stem}.csv"
    pd.DataFrame({
        "y_true": all_labels,
        "y_prob": all_probs,
        "y_pred": all_preds
    }).to_csv(pred_csv, index=False)

    print(f"[INFO] Saved predictions → {pred_csv}")


if __name__ == "__main__":
    main()
