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
from training_utils import evaluate     # <-- your evaluate()

# --------------------------------------------------------------------
# Parse CLI arguments
# --------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run saved GNN model on data")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Checkpoint file (e.g. best_model.pt)")

    parser.add_argument("--data_file", type=str, required=True,
                        help="Dataset file (.csv or .npz) to evaluate")

    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config YAML")

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


# --------------------------------------------------------------------
# Load dataset dynamically from .csv or .npz
# --------------------------------------------------------------------
def load_dataset(data_file, selected_features, scaler=None):
    p = Path(data_file)

    if p.suffix not in {".csv", ".npz"}:
        raise ValueError("data_file must end in .csv or .npz")

    # Try to extract loop order (your naming convention: *_7.csv)
    digits = "".join(c for c in p.stem if c.isdigit())
    if digits == "":
        raise ValueError("Cannot extract loop order from filename. "
                         "Use naming like den_graph_data_7.csv")

    loop_order = int(digits)

    ds, scaler, feats = create_simple_dataset(
        file_ext=loop_order,
        selected_features=selected_features,
        normalize=True,
        data_dir=str(p.parent),
        scaler=scaler,
        n_jobs=4,
        chunk_size=1000
    )
    return ds, scaler


# --------------------------------------------------------------------
# Main evaluation script
# --------------------------------------------------------------------
def main():
    args = parse_args()

    # ----------------------------
    # Load config
    # ----------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    features_cfg = cfg["features"]
    experiment_cfg = cfg["experiment"]
    selected_features = features_cfg["selected_features"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Dataset
    # ----------------------------
    print(f"Loading dataset from: {args.data_file}")
    dataset, _ = load_dataset(
        data_file=args.data_file,
        selected_features=selected_features
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    num_features = dataset[0].x.shape[1]

    # ----------------------------
    # Model
    # ----------------------------
    model = create_gnn_model(
        architecture=model_cfg["name"],
        num_features=num_features,
        hidden_dim=model_cfg["hidden_channels"],
        num_classes=1,
        dropout=model_cfg["dropout"],
        num_layers=model_cfg["num_layers"]
    ).to(device)

    # ----------------------------
    # Load checkpoint
    # ----------------------------
    checkpoint_path = Path(experiment_cfg["model_dir"]) / args.model_name
    print(f"Loading model checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # ----------------------------
    # Evaluate
    # ----------------------------
    print("\n=== MODEL EVALUATION ===")

    avg_loss, accuracy, metrics = evaluate(
        model, loader, device=device,
        threshold=args.threshold,
        log_threshold_curves=True,   # you can turn off if you want
        split_name="eval"
    )

    # Print summary
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

    # ----------------------------
    # Save predictions
    # ----------------------------
    # evaluate() stores everything inside metrics? No.
    # We reconstruct them directly from batch-level computation:
    # We have all_probs inside evaluate(), but evaluate() does not return them.
    # So: we recompute for saving.

    print("\nRecomputing probabilities for saving predictions…")

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
    all_labels = np.concatenate(all_labels)

    output_csv = f"predictions_{Path(args.data_file).stem}.csv"
    df = pd.DataFrame({
        "y_true": all_labels,
        "y_prob": all_probs,
        "y_pred": (all_probs > args.threshold).astype(int)
    })
    df.to_csv(output_csv, index=False)

    print(f"\nSaved predictions → {output_csv}")


if __name__ == "__main__":
    main()
