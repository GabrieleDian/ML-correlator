import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wandb
from types import SimpleNamespace
import matplotlib.pyplot as plt
from GNN_architectures import create_gnn_model


def train_epoch(model, train_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, loader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate_feature_importance(model, dataset, feature_names, device):
    """Evaluate which features are most important for the model."""
    model.eval()
    feature_gradients = []
    
    with torch.enable_grad():
        for data in dataset:
            data = data.to(device)
            data.x.requires_grad = True
            
            # Forward pass
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y.unsqueeze(0))
            
            # Backward pass
            loss.backward()
            
            # Store gradients
            grad = data.x.grad.abs().mean(dim=0)
            feature_gradients.append(grad.cpu().numpy())
    
    # Average gradients across all graphs
    avg_gradients = np.mean(feature_gradients, axis=0)
    
    # Sort features by importance
    importance_scores = list(zip(feature_names, avg_gradients))
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    return importance_scores


def train_single_fold(model, train_dataset, val_dataset, config, fold_idx, device):
    """Train model for a single fold."""
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    if hasattr(config, 'scheduler_type') and config.scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, epochs=config.epochs, 
            steps_per_epoch=len(train_loader)
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
        )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = config.early_stop_patience if hasattr(config, 'early_stop_patience') else 20
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        if hasattr(config, 'scheduler_type') and config.scheduler_type == 'onecycle':
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Log to WandB
        if wandb.run is not None:
            wandb.log({
                f'fold_{fold_idx}/train_loss': train_loss,
                f'fold_{fold_idx}/val_loss': val_loss,
                f'fold_{fold_idx}/val_accuracy': val_acc,
                f'fold_{fold_idx}/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1}/{config.epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")
    
    return model, train_losses, val_losses, val_accuracies


def train(config,dataset):
    """Main training function with k-fold cross validation and WandB integration."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize WandB
    if hasattr(config, 'use_wandb') and config.use_wandb:
        wandb.init(
            project=config.project,
            config=config.__dict__,
            name=f"{config.model_name}_{config.experiment_name}" if hasattr(config, 'experiment_name') else config.model_name
        )
        
    labels = [data.y.item() for data in dataset]
    
    # K-fold cross validation
    k_folds = config.k_folds if hasattr(config, 'k_folds') else 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_feature_importance = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        
        # Split dataset
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        # Initialize model
        model = create_gnn_model(
            config.model_name,
            num_features=config.in_channels,
            hidden_dim=config.hidden_channels,
            num_classes=2,  # Binary classification
            dropout=config.dropout,
            num_layers=config.num_layers if hasattr(config, 'num_layers') else 3
        ).to(device)
        
        # Train model for this fold
        model, train_losses, val_losses, val_accuracies = train_single_fold(
            model, train_dataset, test_dataset, config, fold, device
        )
        
        # Final evaluation on test set
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        fold_accuracies.append(test_accuracy)
        
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Feature importance for this fold
        if hasattr(dataset[0], 'feature_names'):
            importance = evaluate_feature_importance(
                model, test_dataset, dataset[0].feature_names, device
            )
            fold_feature_importance.append(importance)
        
        # Log fold results to WandB
        if wandb.run is not None:
            wandb.log({
                f'fold_{fold}/test_accuracy': test_accuracy,
                f'fold_{fold}/test_loss': test_loss,
                'fold': fold
            })
    
    # Calculate and log final results
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\n{'='*60}")
    print(f"Cross-validation Results:")
    print(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  All Folds: {fold_accuracies}")
    
    # Average feature importance across folds
    avg_importance = {}
    if fold_feature_importance and hasattr(dataset[0], 'feature_names'):
        for feat_name in dataset[0].feature_names:
            scores = []
            for fold_imp in fold_feature_importance:
                for feat, score in fold_imp:
                    if feat == feat_name:
                        scores.append(score)
                        break
            avg_importance[feat_name] = np.mean(scores) if scores else 0
        
        print(f"\nTop 10 Most Important Features:")
        sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, score) in enumerate(sorted_importance[:10]):
            print(f"  {i+1}. {feat:25s}: {score:.4f}")
    
    # Log final results to WandB
    if wandb.run is not None:
        wandb.log({
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'fold_accuracies': fold_accuracies
        })
        
        # Log feature importance as a bar chart
        if avg_importance:
            feature_names = list(avg_importance.keys())[:20]  # Top 20 features
            importance_scores = [avg_importance[f] for f in feature_names]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_names)), importance_scores)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Importance Score')
            plt.title('Feature Importance')
            plt.tight_layout()
            wandb.log({"feature_importance": wandb.Image(plt)})
            plt.close()
        
        # Finish WandB run
        wandb.finish()
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_accuracies': fold_accuracies,
        'feature_importance': avg_importance
    }


def compare_configurations(configs, dataset_generator_fn):
    """Compare multiple configurations and return results."""
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config_name}")
        print(f"{'='*60}")
        
        # Generate dataset with the configuration
        dataset, scaler = dataset_generator_fn(config)
        config.dataset = dataset
        config.in_channels = dataset[0].x.shape[1]
        
        # Train and evaluate
        result = train(config)
        results[config_name] = result
    
    # Visualize comparison
    if len(results) > 1:
        plot_configuration_comparison(results)
    
    return results


def plot_configuration_comparison(results):
    """Plot comparison of different configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    configs = list(results.keys())
    means = [results[c]['mean_accuracy'] for c in configs]
    stds = [results[c]['std_accuracy'] for c in configs]
    
    x = np.arange(len(configs))
    ax1.bar(x, means, yerr=stds, capsize=10)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)
    
    # Feature importance comparison (if available)
    if 'feature_importance' in results[configs[0]] and results[configs[0]]['feature_importance']:
        # Get top features from first configuration
        first_config_imp = results[configs[0]]['feature_importance']
        top_features = sorted(first_config_imp.items(), key=lambda x: x[1], reverse=True)[:5]
        feature_names = [f[0] for f in top_features]
        
        # Compare importance across configurations
        for i, feat in enumerate(feature_names):
            scores = [results[c]['feature_importance'].get(feat, 0) for c in configs]
            ax2.plot(configs, scores, marker='o', label=feat)
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Top Feature Importance Across Configurations')
        ax2.legend()
        ax2.set_xticklabels(configs, rotation=45)
    
    plt.tight_layout()
    
    if wandb.run is not None:
        wandb.log({"configuration_comparison": wandb.Image(plt)})
    
    plt.show()