import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from GNN_architectures import create_gnn_model
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Add confusion matrix for debugging
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1,zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'confusion_matrix': cm  
    }

def train_epoch(model, train_loader, optimizer, device, scheduler=None, pos_weight= None, scheduler_type=None, threshold=0.5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out.view(-1), batch.y.float())

        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(out.view(-1))
        pred = (probs > threshold).long()
        all_probs.append(probs.detach().cpu())
        all_preds.append(pred)
        all_labels.append(batch.y.float())

        correct += pred.eq(batch.y.float()).sum().item()
        total += batch.y.size(0)
        total_loss += loss.item() * batch.y.size(0)  # Weight by batch size

        # Step scheduler if using OneCycleLR
        if scheduler_type == 'onecycle' and scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / total  # Proper average
    accuracy = correct / total
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['roc_auc'] = roc_auc_score(all_labels.cpu().numpy(), all_probs.cpu().numpy())
    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels.cpu().numpy(),  all_probs.cpu().numpy())
    pr_auc_val = auc(recall_vals, precision_vals)
    metrics['pr_auc'] = pr_auc_val


    return avg_loss, accuracy, metrics

def evaluate(model, test_loader, device, pos_weight=None, threshold=0.5, log_threshold_curves=True):
    """Evaluate model on test set, optionally sweeping thresholds"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out.view(-1), batch.y.float())
            
            probs = torch.sigmoid(out.view(-1))
            pred = (probs > threshold).long()
            all_probs.append(probs.detach().cpu())
            all_preds.append(pred)
            all_labels.append(batch.y)
            
            correct += pred.eq(batch.y.float()).sum().item()
            total += batch.y.size(0)
            total_loss += loss.item() * batch.y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['roc_auc'] = roc_auc_score(all_labels.cpu().numpy(), all_probs.cpu().numpy())
    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels.cpu().numpy(),  all_probs.cpu().numpy())
    pr_auc_val = auc(recall_vals, precision_vals)
    metrics['pr_auc'] = pr_auc_val
    
    # ---- Threshold sweep ----
    if log_threshold_curves:
        thresholds = np.linspace(0, 1, 50)
        accs, precs, recs = [], [], []
        y_true = all_labels.numpy()
        y_prob = all_probs.numpy()
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            accs.append((y_pred == y_true).mean())
            precs.append(precision_score(y_true, y_pred, zero_division=0))
            recs.append(recall_score(y_true, y_pred, zero_division=0))
        metrics['thresholds'] = thresholds
        metrics['accuracy_vs_threshold'] = accs
        metrics['precision_vs_threshold'] = precs
        metrics['recall_vs_threshold'] = recs
        
    
    return avg_loss, accuracy, metrics




def train(config, train_dataset, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases if configured
    if getattr(config, 'use_wandb', False):
        wandb.init(
            project=config.project,
            name=getattr(config, 'experiment_name', config.model_name),
            config=config.__dict__
        )
    # Train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = create_gnn_model(
        config.model_name,
        num_features=config.in_channels,
        hidden_dim=config.hidden_channels,
        num_classes=1,  # Binary classification
        dropout=config.dropout,
        num_layers=getattr(config, 'num_layers', 3)
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    
    # Configure scheduler
    scheduler_type = getattr(config, 'scheduler_type', None)
    if scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.lr * 3,  # Peak at 3x base lr
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Spend 30% of time in warmup
            anneal_strategy='cos',
            div_factor=25.0,  # Start at lr/25
            final_div_factor=1000.0  # End at lr/1000
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
        )
    else:
        scheduler = None

    print("\nStarting training...")
    print(f"Model architecture: {config.model_name}")
    print(f"Hidden dim: {config.hidden_channels}, Layers: {getattr(config, 'num_layers', 3)}")
    print(f"Initial LR: {optimizer.param_groups[0]['lr']}")
    
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(config.epochs):
        # Training
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            threshold = config.threshold,
        )
        
        # Test
        test_loss, test_acc, test_metrics = evaluate(model, test_loader, device, threshold=config.threshold, log_threshold_curves=config.log_threshold_curves)
        
        # Step plateau scheduler after epoch
        if scheduler_type == 'plateau' and scheduler is not None:
            scheduler.step(test_loss)

        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
    
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        if wandb.run is not None:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_roc_auc': train_metrics['roc_auc'],
                'train_pr_auc': train_metrics['pr_auc'],
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_roc_auc': test_metrics['roc_auc'],
                'test_pr_auc': test_metrics['pr_auc']

            })

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}, "
                  f"LR={current_lr:.6f}")            
            # Print confusion matrix for debugging
            #print(f"Train Confusion Matrix:\n{train_metrics['confusion_matrix']}")
            #print(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")

 
    if wandb.run is not None and config.log_threshold_curves and 'thresholds' in test_metrics:
        thresholds = test_metrics['thresholds']
        accs = test_metrics['accuracy_vs_threshold']
        precs = test_metrics['precision_vs_threshold']
        recs = test_metrics['recall_vs_threshold']

        # Create matplotlib plot
        plt.figure()
        plt.plot(thresholds, accs, label="Accuracy")
        plt.plot(thresholds, precs, label="Precision")
        plt.plot(thresholds, recs, label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.title(f"Threshold curves - train:{config.train_loop_order}, test:{config.test_loop_order}")
        wandb.log({"threshold_curves": wandb.Image(plt)})
        plt.close()
        wandb.finish()

    return {
        'model_state': best_model_state,
        'final_train_acc': train_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'final_train_roc_auc': train_metrics['roc_auc'],
        'final_train_pr_auc': train_metrics['pr_auc'],
        'final_test_roc_auc': test_metrics['roc_auc'],
        'final_test_pr_auc': test_metrics['pr_auc']
    }
