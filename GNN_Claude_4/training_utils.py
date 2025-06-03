import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from GNN_architectures import create_gnn_model
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Add confusion matrix for debugging
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': cm
    }

def train_epoch(model, train_loader, optimizer, device, scheduler=None, scheduler_type=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        all_preds.append(pred)
        all_labels.append(batch.y)

        correct += pred.eq(batch.y).sum().item()
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
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, accuracy, metrics

def evaluate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            
            pred = out.argmax(dim=1)
            all_preds.append(pred)
            all_labels.append(batch.y)
            
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
            total_loss += loss.item() * batch.y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, accuracy, metrics

def train(config, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split dataset into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {train_size}, Val size: {val_size}")

    if getattr(config, 'use_wandb', False):
        wandb.init(
            project=config.project,
            name=getattr(config, 'experiment_name', config.model_name),
            config=config.__dict__
        )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = create_gnn_model(
        config.model_name,
        num_features=config.in_channels,
        hidden_dim=config.hidden_channels,
        num_classes=2,
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
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(config.epochs):
        # Training
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler,
            scheduler_type=scheduler_type
        )
        
        # Validation
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, device)
        
        # Step plateau scheduler after epoch
        if scheduler_type == 'plateau' and scheduler is not None:
            scheduler.step(val_loss)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
                'train_f1': train_metrics['f1'],
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'current_lr': current_lr
            })

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, "
                  f"LR={current_lr:.6f}")
            
            # Print confusion matrix for debugging
            #print(f"Train Confusion Matrix:\n{train_metrics['confusion_matrix']}")
            #print(f"Val Confusion Matrix:\n{val_metrics['confusion_matrix']}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    if wandb.run is not None:
        wandb.finish()

    return {
        'model_state': best_model_state,
        'final_train_acc': train_acc,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }