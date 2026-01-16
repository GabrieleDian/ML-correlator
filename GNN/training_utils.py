import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import wandb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
from GNN_architectures import create_gnn_model
import numpy as np
import matplotlib.pyplot as plt
import os, time


# ==============================================================
# Basic metrics helper, ansatz fraction calculator
# ==============================================================

def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": cm,
    }

def compute_threshold_neg_removal_fraction(labels, probs, min_pos_prob):
    """Compute fraction of true-0 graphs with prob < min_pos_prob.

    Denominator is the number of true-0 graphs in `labels`.
    Returns 0.0 if min_pos_prob is None or there are no negatives.
    """
    if min_pos_prob is None:
        return 0.0
    labels = labels.detach().cpu().view(-1)
    probs = probs.detach().cpu().view(-1)
    n_neg = int((labels == 0).sum().item())
    if n_neg == 0:
        return 0.0
    removable_negs = int(((labels == 0) & (probs < float(min_pos_prob))).sum().item())
    return removable_negs / n_neg


def compute_safe_ansatz_fraction(labels, probs):
    """Backward-compatible wrapper.

    Training split: compute threshold from THIS split (min prob among true positives)
    and then compute fraction of true negatives below that threshold.
    """
    labels = labels.detach().cpu().view(-1)
    probs = probs.detach().cpu().view(-1)

    true_one_probs = probs[labels == 1]
    if true_one_probs.numel() == 0:
        return 0.0

    t_min = float(true_one_probs.min().item())
    return compute_threshold_neg_removal_fraction(labels, probs, t_min)


# New helper: return minimum predicted probability among true positives (or None)
def compute_min_positive_prob(labels, probs):
    """Return minimum predicted probability among true positives, or None if no positives."""
    labels = labels.detach().cpu().view(-1)
    probs = probs.detach().cpu().view(-1)
    true_one_probs = probs[labels == 1]
    if true_one_probs.numel() == 0:
        return None
    return float(true_one_probs.min().item())

# ==============================================================
# Training for one epoch
# ==============================================================

def train_epoch(model, train_loader, optimizer, device,
                scheduler=None, pos_weight=None,
                scheduler_type=None, threshold=0.5):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []

    # pos_weight is a scalar applied to positive examples in BCEWithLogitsLoss
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor([float(pos_weight)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw) if pw is not None else nn.BCEWithLogitsLoss()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out.view(-1), batch.y.float())
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Scheduler (OneCycle)
        if scheduler_type == "onecycle" and scheduler is not None:
            scheduler.step()

        probs = torch.sigmoid(out.view(-1))
        preds = (probs > threshold).long()

        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(batch.y.detach().cpu().float())

        correct += preds.eq(batch.y.float()).sum().item()
        total += batch.y.size(0)

        total_loss += loss.item() * batch.y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)


    metrics = compute_metrics(all_labels, all_preds)
   # Move to CPU before converting to numpy
    labels_np = all_labels.detach().cpu().numpy()
    probs_np = all_probs.detach().cpu().numpy()

    metrics["roc_auc"] = roc_auc_score(labels_np, probs_np)
    prec, rec, _ = precision_recall_curve(labels_np, probs_np)
    metrics["pr_auc"] = auc(rec, prec)

    # Minimum predicted probability among true positives (TRAIN, per epoch)
    min_pos = compute_min_positive_prob(all_labels, all_probs)
    metrics["lowest_prob_true1"] = min_pos
    metrics["lowest_prob_true1_train"] = min_pos
    metrics["train_lowest_prob_true1"] = min_pos

    # Training-set neg removal fraction uses training-set min_pos
    nrf = compute_threshold_neg_removal_fraction(all_labels, all_probs, min_pos)
    metrics["neg_removal_fraction"] = nrf

    return avg_loss, accuracy, metrics


# ==============================================================
# Unified evaluation (for validation or test)
# ==============================================================

def evaluate(model, loader, device, pos_weight=None,
             threshold=0.5, log_threshold_curves=False, split_name="val",
             ref_min_pos_prob=None):
    # --------------------------------------------------------------
    # EARLY EXIT: When test set is skipped or loader is None
    # --------------------------------------------------------------
    if loader is None:
        return 0.0, 0.0, {
            "roc_auc": None,
            "pr_auc": None,
            "recall": None,
            "accuracy": None,
            "neg_removal_fraction": None,
        }

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []

    # pos_weight is a scalar applied to positive examples in BCEWithLogitsLoss
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor([float(pos_weight)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw) if pw is not None else nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out.view(-1), batch.y.float())

            probs = torch.sigmoid(out.view(-1))
            preds = (probs > threshold).long()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.float().cpu().float())
        
            correct += preds.eq(batch.y.float()).sum().item()
            total += batch.y.size(0)
            total_loss += loss.item() * batch.y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_labels, all_preds)
   # Move to CPU before converting to numpy
    labels_np = all_labels.detach().cpu().numpy()
    probs_np = all_probs.detach().cpu().numpy()

    metrics["roc_auc"] = roc_auc_score(labels_np, probs_np)
    prec, rec, _ = precision_recall_curve(labels_np, probs_np)
    metrics["pr_auc"] = auc(rec, prec)

    # Minimum predicted probability among true positives on THIS split
    split_min_pos = compute_min_positive_prob(all_labels, all_probs)
    metrics["lowest_prob_true1"] = split_min_pos
    if split_name:
        metrics[f"{split_name}_lowest_prob_true1"] = split_min_pos

    # Compute neg removal fraction.
    # For val/test, pass ref_min_pos_prob=train_min_pos_prob to use training-derived threshold.
    if ref_min_pos_prob is None:
        ref_min_pos_prob = split_min_pos

    metrics["neg_removal_fraction"] = compute_threshold_neg_removal_fraction(all_labels, all_probs, ref_min_pos_prob)

    return avg_loss, accuracy, metrics


# ==============================================================
# Threshold-based evaluation using training minimum probability
# ==============================================================

def evaluate_threshold_from_train(train_min_prob, test_labels, test_probs):
    """
    Uses the minimum probability from true positives in training set as a threshold
    on the test set to evaluate false negative guarantee and negative coverage.
    
    Args:
        train_min_prob: minimum probability assigned to true positives in training set
        test_labels: tensor of true labels (0 or 1) on test set
        test_probs: tensor of predicted probabilities on test set
    
    Returns:
        dict with results and prints summary
    """
    test_labels = test_labels.detach().cpu()
    test_probs = test_probs.detach().cpu()
    
    # Apply threshold
    test_preds = (test_probs >= train_min_prob).long()
    
    # Get true positives and false negatives
    true_positives = ((test_preds == 1) & (test_labels == 1)).sum().item()
    false_negatives = ((test_preds == 0) & (test_labels == 1)).sum().item()
    
    # Check no false negatives
    no_false_negatives = (false_negatives == 0)
    
    # Negatives with probability below threshold
    true_negatives_below = ((test_probs < train_min_prob) & (test_labels == 0)).sum().item()
    pct_negatives_below = 100.0 * true_negatives_below / len(test_labels)
    
    print(f"\n{'='*60}")
    print(f"Threshold-based evaluation (threshold={train_min_prob:.4f} from training set)")
    print(f"{'='*60}")
    print(f"1) False Negatives Guarantee: {'✓ NO FALSE NEGATIVES' if no_false_negatives else '✗ FALSE NEGATIVES EXIST'}")
    print(f"   - True Positives: {true_positives}")
    print(f"   - False Negatives: {false_negatives}")
    print(f"2) Negatives below threshold: {true_negatives_below}/{len(test_labels)} ({pct_negatives_below:.2f}%)")
    print(f"{'='*60}\n")
    
    return {
        "threshold": float(train_min_prob),
        "no_false_negatives": no_false_negatives,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "negatives_below_threshold": true_negatives_below,
        "pct_negatives_below_threshold": pct_negatives_below
    }
    

# ==============================================================
# Main training loop
# ==============================================================

def train(config, train_dataset, val_dataset, test_dataset, use_wandb=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.time()

    # Pull pos_weight from config if present
    pos_weight = getattr(config, "pos_weight", None)
    if pos_weight is not None:
        print(f"Using pos_weight={pos_weight}")

    # Initialize wandb
    if 'WANDB_SWEEP_ID' in os.environ or getattr(config, 'use_wandb', False):
        wandb.init(
            project=getattr(config, 'wandb_project', getattr(config, 'project', 'GNN-train')),
            config=config.__dict__,
            reinit=True
        )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)if test_dataset is not None else None


    import itertools
    from sklearn.preprocessing import StandardScaler

    # --- helpers to iterate graphs out of ConcatDataset or plain lists ---
    def iter_graphs(ds):
        if ds is None:
            return []
        if isinstance(ds, torch.utils.data.ConcatDataset):
            return itertools.chain.from_iterable(ds.datasets)
        return iter(ds)

    # 1) Find the single GLOBAL feature width across all splits
    all_ds = [train_dataset]
    if val_dataset is not None:
        all_ds.append(val_dataset)
    if test_dataset is not None:
        all_ds.append(test_dataset)

    global_max_features = 0
    for ds in all_ds:
        for g in iter_graphs(ds):
            global_max_features = max(global_max_features, g.x.shape[1])

    # 2) PAD every graph in every split up to that width (never truncate)
    for ds in all_ds:
        for g in iter_graphs(ds):
            n, f = g.x.shape
            if f < global_max_features:
                pad = torch.zeros(n, global_max_features - f, dtype=g.x.dtype, device=g.x.device)
                g.x = torch.cat([g.x, pad], dim=1)

    # 3) (Re-)NORMALIZE consistently after padding (fit on train, apply to all)
    #    If you already normalized inside create_simple_dataset, this will just refit and reapply.
    train_feats = torch.cat([g.x for g in iter_graphs(train_dataset)], dim=0).cpu().numpy()
    scaler = StandardScaler().fit(train_feats)

    for ds in all_ds:
        for g in iter_graphs(ds):
            g.x = torch.as_tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32, device=g.x.device)

    # 4) Sanity checks: after padding+norm, every split MUST have one width = global_max_features
    def widths(ds):
        ws = sorted({g.x.shape[1] for g in iter_graphs(ds)})
        return ws


    # 5) Build the model with the TRUE input dimension
    true_in_channels = global_max_features
    print(f"Detected input feature dimension for model: {true_in_channels}")
    
    model = create_gnn_model(
        config.model_name,
        num_features=true_in_channels,   # <<< key change: use detected width
        hidden_dim=config.hidden_channels,
        num_classes=1,
        dropout=config.dropout,
        num_layers=getattr(config, "num_layers", 3)
    ).to(device)


    # Trainable parameter count (re-exposed)
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Model has {num_params} trainable parameters")

    # If W&B is enabled, log static hyperparams once
    if use_wandb and wandb.run is not None:
        wandb.log({
            "number_of_parameters": num_params,
            "pos_weight": pos_weight,
        })

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    # Scheduler setup
    scheduler_type = getattr(config, "scheduler_type", None)
    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate * 3,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    else:
        scheduler = None

    print("\nStarting training...")
    print(f"Model: {config.model_name}, Hidden={config.hidden_channels}, Layers={getattr(config, 'num_layers', 3)}")
    print(f"Initial LR: {optimizer.param_groups[0]['lr']}")

    best_val_loss = float("inf")
    best_state = None

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------
    for epoch in range(config.epochs):
        # ---- Training ----
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler, scheduler_type=scheduler_type,
            threshold=config.threshold,
            pos_weight=pos_weight,
        )

        train_min_pos_prob = train_metrics.get("lowest_prob_true1")

        # ---- Validation ----
        if val_dataset is not None:
            val_loss, val_acc, val_metrics = evaluate(
                model, val_loader, device,
                threshold=config.threshold,
                log_threshold_curves=False,
                split_name="val",
                ref_min_pos_prob=train_min_pos_prob,
                pos_weight=pos_weight,
            )

            # Step LR scheduler (plateau only after val)
            if scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(val_loss)

            # Track best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()

            if epoch % 10 == 0 or epoch == config.epochs - 1:
                print(f"Epoch {epoch:3d}/{config.epochs}: "
                      f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            # ---- W&B logging ----
            if use_wandb and wandb.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_pr_auc": train_metrics.get("pr_auc"),
                    "train_roc_auc": train_metrics.get("roc_auc"),
                    "train_recall": train_metrics.get("recall"),
                    "train_neg_removal_fraction": train_metrics.get("neg_removal_fraction"),
                    "train_lowest_prob_true1": train_metrics.get("train_lowest_prob_true1"),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_pr_auc": val_metrics.get("pr_auc"),
                    "val_roc_auc": val_metrics.get("roc_auc"),
                    "val_recall": val_metrics.get("recall"),
                    "val_neg_removal_fraction": val_metrics.get("neg_removal_fraction"),
                    "val_lowest_prob_true1": val_metrics.get("val_lowest_prob_true1"),
                    "lr": optimizer.param_groups[0]["lr"],
                    "in_channels": true_in_channels,
                }
                wandb.log(log_dict, step=epoch)

        # ---- No validation case ----
        else:
            # If no validation set, just step scheduler (if not OneCycle)
            if scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(train_loss)

            if epoch % 10 == 0 or epoch == config.epochs - 1:
                print(f"Epoch {epoch:3d}/{config.epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

            # Track best model by training loss only
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state = model.state_dict().copy()

            # W&B logging
            if use_wandb and wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_pr_auc": train_metrics.get("pr_auc"),
                    "train_roc_auc": train_metrics.get("roc_auc"),
                    "train_recall": train_metrics.get("recall"),
                    "train_neg_removal_fraction": train_metrics.get("neg_removal_fraction"),
                    "train_lowest_prob_true1": train_metrics.get("train_lowest_prob_true1"),
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=epoch)

    # --------------------------------------------------
    # AFTER TRAINING: compute test metrics
    # --------------------------------------------------
    model.load_state_dict(best_state)

    # Get training-derived threshold from the last epoch's metrics (or best epoch if you later store it)
    train_min_pos_prob = train_metrics.get("lowest_prob_true1")

    # Evaluate on test set
    if test_loader is not None:
        test_loss, test_acc, test_metrics = evaluate(
            model, test_loader, device,
            threshold=config.threshold,
            log_threshold_curves=config.log_threshold_curves,
            split_name="test",
            ref_min_pos_prob=train_min_pos_prob,
            pos_weight=pos_weight,
        )

        # expose final test-set minimum prob among true 1s
        test_metrics["test_lowest_prob_true1"] = test_metrics.get("lowest_prob_true1")

        # Evaluate using training-set threshold to guarantee no false negatives
        if train_min_pos_prob is not None:
            # Collect test labels and probabilities for threshold evaluation
            test_all_probs, test_all_labels = [], []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    probs = torch.sigmoid(out.view(-1))
                    test_all_probs.append(probs.cpu())
                    test_all_labels.append(batch.y.float().cpu())

            test_all_probs = torch.cat(test_all_probs)
            test_all_labels = torch.cat(test_all_labels)

            # Apply threshold from training set
            test_metrics["threshold_eval"] = evaluate_threshold_from_train(
                train_min_pos_prob,
                test_all_labels,
                test_all_probs
            )
    else:
        test_loss, test_acc, test_metrics = None, None, {
            "pr_auc": None,
            "roc_auc": None,
            "recall": None,
            "neg_removal_fraction": None,
        }

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s.")

    # End-of-training predictive model checks
    train_min = train_metrics.get("train_lowest_prob_true1")

    predictive_model_val = None
    predictive_model_test = None

    if val_dataset is not None:
        val_min = None
        if 'val_metrics' in locals() and isinstance(locals().get('val_metrics'), dict):
            val_min = val_metrics.get("val_lowest_prob_true1")
        if train_min is not None and val_min is not None:
            predictive_model_val = bool(train_min <= val_min)

    # Only compute test predictive metric when validation is disabled and test exists
    if val_dataset is None and test_loader is not None:
        test_min = test_metrics.get("test_lowest_prob_true1")
        if train_min is not None and test_min is not None:
            predictive_model_test = bool(train_min <= test_min)

    # Print a small end-of-run summary (avoid legacy key names)
    print("\n=== Summary ===")
    print(f"Train neg-removal fraction: {train_metrics.get('neg_removal_fraction', None)}")

    if val_dataset is not None:
        print(f"Val   neg-removal fraction: {val_metrics.get('neg_removal_fraction', None) if 'val_metrics' in locals() else None}")

    if test_loader is not None:
        print(f"Test  neg-removal fraction: {test_metrics.get('neg_removal_fraction', None)}")

    if val_dataset is not None:
        print(f"Predictive model (val): {predictive_model_val}")
    elif test_loader is not None:
        print(f"Predictive model (test): {predictive_model_test}")

    # Final Test metrics logging
    if use_wandb and wandb.run is not None:
        # Log predictive model flags once at end
        end_flags = {}
        if predictive_model_val is not None:
            end_flags["predictive_model"] = predictive_model_val
        if predictive_model_test is not None:
            end_flags["predictive_model_test"] = predictive_model_test
        if end_flags:
            wandb.log(end_flags)

    # Optional W&B threshold curves
    if wandb.run is not None and config.log_threshold_curves and "thresholds" in test_metrics:
        plt.figure()
        plt.plot(test_metrics["thresholds"], test_metrics["accuracy_vs_threshold"], label="Accuracy")
        plt.plot(test_metrics["thresholds"], test_metrics["precision_vs_threshold"], label="Precision")
        plt.plot(test_metrics["thresholds"], test_metrics["recall_vs_threshold"], label="Recall")
        plt.xlabel("Threshold"); plt.ylabel("Score"); plt.legend()
        plt.title(f"Threshold curves ({config.train_loop_order}→{config.test_loop_order})")
        wandb.log({"threshold_curves": wandb.Image(plt)})
        plt.close()

    results = {
        "model_state": model.state_dict().copy(),

        # --- TRAIN metrics ---
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_pr_auc": train_metrics.get("pr_auc"),
        "train_roc_auc": train_metrics.get("roc_auc"),
        "train_recall": train_metrics.get("recall"),
        "train_neg_removal_fraction": train_metrics.get("neg_removal_fraction"),
        "train_lowest_prob_true1": train_metrics.get("train_lowest_prob_true1"),

        # --- VAL metrics (if available) ---
        "val_loss": val_loss if val_dataset is not None else None,
        "val_acc": val_acc if val_dataset is not None else None,
        "val_pr_auc": val_metrics.get("pr_auc") if val_dataset is not None else None,
        "val_roc_auc": val_metrics.get("roc_auc") if val_dataset is not None else None,
        "val_recall": val_metrics.get("recall") if val_dataset is not None else None,
        "val_neg_removal_fraction": val_metrics.get("neg_removal_fraction") if val_dataset is not None else None,
        "val_lowest_prob_true1": val_metrics.get("val_lowest_prob_true1") if val_dataset is not None else None,

        # --- TEST metrics ---
        "test_loss": test_loss if test_loader is not None else None,
        "test_acc": test_acc if test_loader is not None else None,
        "test_pr_auc": test_metrics.get("pr_auc") if test_loader is not None else None,
        "test_roc_auc": test_metrics.get("roc_auc") if test_loader is not None else None,
        "test_recall": test_metrics.get("recall") if test_loader is not None else None,
        "test_neg_removal_fraction": (test_metrics.get("neg_removal_fraction") if test_loader is not None else None),
        "test_lowest_prob_true1": (test_metrics.get("test_lowest_prob_true1") if test_loader is not None else None),

        # --- Predictive model flags ---
        "predictive_model": predictive_model_val,
        "predictive_model_test": predictive_model_test,

        # --- Runtime / model size ---
        "total_time": total_time,
        "number_of_parameters": num_params,
    }
    


    return results

