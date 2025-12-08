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

def compute_safe_ansatz_fraction(labels, probs):
    """
    labels: tensor (N,) of {0,1}, on CPU
    probs: tensor (N,) of predicted probabilities in [0,1], on CPU

    Returns float in [0,1]: fraction of total graphs needed to 
    guarantee all true 1-graphs are included.
    """

    labels = labels.detach().cpu()
    probs = probs.detach().cpu()

    # Extract only probabilities of true positives
    true_one_probs = probs[labels == 1]

    # Edge case: if the dataset has zero positive examples
    if true_one_probs.numel() == 0:
        return 0.0

    # Minimum probability among true 1s
    t_min = true_one_probs.min().item()

    # All graphs predicted above this threshold
    selected = (probs >= t_min).sum().item()

    # Fraction of ansatz kept
    return 1-selected / len(probs)

# ==============================================================
# Training for one epoch
# ==============================================================
def train_epoch(model, train_loader, optimizer, device,
                scheduler=None, pos_weight=None,
                scheduler_type=None, threshold=0.5):
    pos_weight = torch.tensor([9.0]).to(device)
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    safe_fraction = compute_safe_ansatz_fraction(all_labels, all_probs)
    metrics["safe_ansatz_fraction"] = safe_fraction

    return avg_loss, accuracy, metrics


# ==============================================================
# Unified evaluation (for validation or test)
# ==============================================================
def evaluate(model, loader, device, pos_weight=None,
             threshold=0.5, log_threshold_curves=False, split_name="val"):
    # --------------------------------------------------------------
    # EARLY EXIT: When test set is skipped or loader is None
    # --------------------------------------------------------------
    if loader is None:
        return 0.0, 0.0, { 
            "roc_auc": None,
            "pr_auc": None,
            "recall": None,
            "accuracy": None,
            "safe_ansatz_fraction": None
        }
    pos_weight = torch.tensor([9.0]).to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    # Optional threshold sweep
    if log_threshold_curves:
        thresholds = np.linspace(0, 1, 50)
        accs, precs, recs = [], [], []
        y_true, y_prob = all_labels.numpy(), all_probs.numpy()
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            accs.append((y_pred == y_true).mean())
            precs.append(precision_score(y_true, y_pred, zero_division=0))
            recs.append(recall_score(y_true, y_pred, zero_division=0))
        metrics["thresholds"] = thresholds
        metrics["accuracy_vs_threshold"] = accs
        metrics["precision_vs_threshold"] = precs
        metrics["recall_vs_threshold"] = recs

   # Comment this out inside evaluate():
# print(f"{split_name.capitalize()} → "
#       f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
#       f"ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")

    # Compute your safe-ansatz metric
    safe_fraction = compute_safe_ansatz_fraction(all_labels, all_probs)
    metrics["safe_ansatz_fraction"] = safe_fraction

    return avg_loss, accuracy, metrics


# ==============================================================
# Main training loop
# ==============================================================
def train(config, train_dataset, val_dataset, test_dataset, use_wandb=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.time()

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


    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")

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
            threshold=config.threshold
        )

        # ---- Validation (only if val_dataset exists) ----
        if val_dataset is not None:
            val_loss, val_acc, val_metrics = evaluate(
                model, val_loader, device,
                threshold=config.threshold,
                log_threshold_curves=False,
                split_name="val"
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
            if use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_pr_auc": train_metrics.get("pr_auc"),
                    "train_roc_auc": train_metrics.get("roc_auc"),
                    "train_recall": train_metrics.get("recall"),
                    "train_safe_ansatz_fraction": train_metrics.get("safe_ansatz_fraction"),
                    "lr": optimizer.param_groups[0]["lr"],
                    "in_channels": true_in_channels
                }

                if val_dataset is not None:
                    log_dict.update({
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_pr_auc": val_metrics.get("pr_auc"),
                        "val_roc_auc": val_metrics.get("roc_auc"),
                        "val_recall": val_metrics.get("recall"),
                        "val_safe_ansatz_fraction": val_metrics.get("safe_ansatz_fraction"),
                    })

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
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_pr_auc": train_metrics["pr_auc"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "train_recall": train_metrics["recall"],
                    "train_safe_ansatz_fraction": train_metrics.get("safe_ansatz_fraction"),
                    "lr": optimizer.param_groups[0]["lr"]
                }, step=epoch)

    # --------------------------------------------------
    # AFTER TRAINING: restore best model
    # --------------------------------------------------
    model.load_state_dict(best_state)

    # Evaluate on test set
    if test_loader is not None:
        test_loss, test_acc, test_metrics = evaluate(
            model, test_loader, device,
            threshold=config.threshold,
            log_threshold_curves=config.log_threshold_curves,
            split_name="test"
    )
    else:
        test_loss, test_acc, test_metrics = None, None, {
            "pr_auc": None,
            "roc_auc": None,
            "recall": None,
            "safe_ansatz_fraction": None
        }


    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s.")

    # Final Test metrics logging
    if use_wandb and wandb.run is not None and test_loader is not None:
        wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_pr_auc": test_metrics.get("pr_auc"),
        "test_roc_auc": test_metrics.get("roc_auc"),
        "test_recall": test_metrics.get("recall"),
        "test_safe_ansatz_fraction": test_metrics.get("safe_ansatz_fraction"),
        "total_time": total_time,
        "number of parameters": num_params})
    elif use_wandb and wandb.run is not None:
        wandb.log({"total_time": total_time})


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
        'train_safe_ansatz_fraction': train_metrics.get("safe_ansatz_fraction"),


        # --- VAL metrics (if available) ---
        "val_loss": val_loss if val_dataset is not None else None,
        "val_acc": val_acc if val_dataset is not None else None,
        "val_pr_auc": val_metrics.get("pr_auc") if val_dataset is not None else None,
        "val_roc_auc": val_metrics.get("roc_auc") if val_dataset is not None else None,
        "val_recall": val_metrics.get("recall") if val_dataset is not None else None,
        'val_safe_ansatz_fraction': val_metrics.get("safe_ansatz_fraction") if val_dataset is not None else None,

        # --- TEST metrics ---
        "test_loss": test_loss if test_loader is not None else None,
        "test_acc": test_acc if test_loader is not None else None,
        "test_pr_auc": test_metrics.get("pr_auc") if test_loader is not None else None,
        "test_roc_auc": test_metrics.get("roc_auc") if test_loader is not None else None,
        "test_recall": test_metrics.get("recall") if test_loader is not None else None,
        "test_safe_ansatz_fraction": (test_metrics.get("safe_ansatz_fraction") if test_loader is not None else None),

        # --- Runtime ---
        "total_time": total_time
        }
    


    return results

