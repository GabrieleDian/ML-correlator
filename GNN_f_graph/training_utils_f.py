import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import wandb
from GNN_architectures_f import create_gnn_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Metric computation
def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics for predicted vs true values.
    """
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def train_epoch(model, train_loader, optimizer, device, scheduler=None, scheduler_type=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    loss_fn = nn.MSELoss()  # or nn.L1Loss()
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # edge_attr included
        out = out.view(-1)  # Flatten to (num_graphs,)
        
        # Compute loss
        loss = loss_fn(out, batch.y.float())
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        # Scheduler step if needed
        if scheduler_type == 'onecycle' and scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * batch.y.size(0)
        all_preds.append(out.detach().cpu())
        all_labels.append(batch.y.float().detach().cpu())
    
    # Concatenate predictions
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(all_labels)
    
    # Compute regression metrics
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_labels)**2)).item()
    
    metrics = {'MAE': mae, 'RMSE': rmse}
    
    return avg_loss, metrics


def evaluate(model, test_loader, device):
    """Evaluate regression model on test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    loss_fn = nn.MSELoss()  # or nn.L1Loss()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            out = out.view(-1)
            loss = loss_fn(out, batch.y.float())
            
            total_loss += loss.item() * batch.y.size(0)
            all_preds.append(out.cpu())
            all_labels.append(batch.y.float().cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(all_labels)
    
    # Regression metrics
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_labels)**2)).item()
    r2 = 1 - torch.sum((all_preds - all_labels)**2) / torch.sum((all_labels - torch.mean(all_labels))**2)
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2.item()}
    
    return avg_loss, metrics



def train(config, train_dataset, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    if getattr(config, 'use_wandb', False):
        wandb.init(
            project=config.project,
            name=getattr(config, 'experiment_name', config.model_name),
            config=config.__dict__
        )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = create_gnn_model(
        config.model_name,
        num_features=config.in_channels,
        hidden_dim=config.hidden_channels,
        num_classes=1,  # Regression output
        dropout=config.dropout,
        num_layers=getattr(config, 'num_layers', 3)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()  # Or nn.L1Loss()

    scheduler_type = getattr(config, 'scheduler_type', None)
    if scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.lr * 3,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
        )
    else:
        scheduler = None

    best_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            loss = loss_fn(out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.y.size(0)

            if scheduler_type == 'onecycle' and scheduler is not None:
                scheduler.step()
        
        train_loss = total_loss / len(train_dataset)
        
        # Evaluation
        test_loss, test_metrics = evaluate(model, test_loader, device)

        # Step plateau scheduler
        if scheduler_type == 'plateau' and scheduler is not None:
            scheduler.step(test_loss)

        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}")
            
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    **{f"test_{k}": v for k,v in test_metrics.items()}
                })
    
    if wandb.run is not None:
        wandb.finish()

    return {
        'model_state': best_model_state,
        'best_test_loss': best_test_loss,
        'best_epoch': best_epoch,
        **{f"test_{k}": v for k,v in test_metrics.items()}
    }
