import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to evaluate the accuracy and other mertics of the model.

def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)  # shape: [batch_size, num_classes]
            preds = out.argmax(dim=1)  # predicted class index
            y_true.append(batch.coeff.cpu())
            y_pred.append(preds.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=None, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average= None, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average= None, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "Number of epochs trained": model.epoch,
    }

    return metrics



def save_model_architecture(model, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename
    i = 1
    while os.path.exists(os.path.join(save_dir, f"model_{i}.txt")):
        i += 1
    file_path = os.path.join(save_dir, f"model_{i}.txt")

    # Save architecture
    with open(file_path, 'w') as f:
        f.write(str(model))

    return file_path  # So you can reuse the path for metrics

def append_evaluation_results(file_path, metrics, loop):
    with open(file_path, 'a') as f:
        if isinstance(loop, list):
            loop_str = ", ".join(map(str, loop))
        else:
            loop_str = str(loop)
        f.write(f"\n\n--- Evaluation Results (loop = {loop_str}) ---\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")