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
# Basic metrics helper
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


# ==============================================================
# Training for one epoch
# ==============================================================
def train_epoch(model, train_loader, optimizer, device,
                scheduler=None, pos_weight=None,
                scheduler_type=None, threshold=0.5):
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
    return avg_loss, accuracy, metrics


# ==============================================================
# Unified evaluation (for validation or test)
# ==============================================================
def evaluate(model, loader, device, pos_weight=None,
             threshold=0.5, log_threshold_curves=False, split_name="val"):
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
            all_labels.append(batch.y.float())

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
    val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    model = create_gnn_model(
        config.model_name,
        num_features=config.in_channels,
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

    for epoch in range(config.epochs):
        # ---- Training ----
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler, scheduler_type=scheduler_type,
            threshold=config.threshold)

        # Step scheduler (plateau only after validation)
        if scheduler_type != "onecycle":
            optimizer.zero_grad()

        # ---- Validation ----
        val_loss, val_acc, val_metrics = evaluate(
            model, val_loader, device,
            threshold=config.threshold,
            log_threshold_curves=False,
            split_name="val"
        )

        if scheduler_type == "plateau" and scheduler is not None:
            scheduler.step(val_loss)

        # ---- W&B logging ----
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_pr_auc": train_metrics["pr_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "lr": optimizer.param_groups[0]["lr"]
            }, step=epoch)

        # ---- Track best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # ---- Testing on best model ----
    model.load_state_dict(best_state)
    test_loss, test_acc, test_metrics = evaluate(
        model, test_loader, device,
        threshold=config.threshold,
        log_threshold_curves=config.log_threshold_curves,
        split_name="test"
    )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s.")

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

    return {
        "model_state": model.state_dict().copy(),
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_test_loss": test_loss,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "final_test_acc": test_acc,
        "final_train_pr_auc": train_metrics["pr_auc"],
        "final_val_pr_auc": val_metrics["pr_auc"],
        "final_test_pr_auc": test_metrics["pr_auc"],
        "final_train_roc_auc": test_metrics["roc_auc"],
        "final_val_roc_auc": val_metrics["roc_auc"],
        "final_test_roc_auc": test_metrics["roc_auc"],
        "final_train_recall": train_metrics["recall"],
        "final_val_recall": val_metrics["recall"],
        "final_test_recall": test_metrics["recall"],
        "total_time": total_time
    }
