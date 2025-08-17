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
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1,zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
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

def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
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
    
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(config.epochs):
        # Training
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler,
            scheduler_type=scheduler_type
        )
        
        # Test
        test_loss, test_acc, test_metrics = evaluate(model, test_loader, device)
        
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
                'train_f1': train_metrics['f1'],
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'current_lr': current_lr
            })

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}, "
                  f"LR={current_lr:.6f}")
            
            # Print confusion matrix for debugging
            #print(f"Train Confusion Matrix:\n{train_metrics['confusion_matrix']}")
            #print(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    print(f"\nBest test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")

    if wandb.run is not None:
        wandb.finish()

    return {
        'model_state': best_model_state,
        'final_train_acc': train_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch
    }