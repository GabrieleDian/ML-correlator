import yaml
import torch
from pathlib import Path
import pandas as pd
from torch_geometric.loader import DataLoader
from graph_builder import  create_simple_dataset  # adjust import
from GNN_architectures import create_gnn_model  # adjust import

# ----------------------------
# 1. Load config
# ----------------------------
config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Model parameters
model_cfg = cfg['model']
features_cfg = cfg['features']
data_cfg = cfg['data']
experiment_cfg = cfg['experiment']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Recreate model
# ----------------------------
# num_features will be determined after creating the dataset
# But for GIN, hidden_channels, num_layers, dropout are from config
hidden_dim = model_cfg['hidden_channels']
num_layers = model_cfg['num_layers']
dropout = model_cfg['dropout']

# ----------------------------
# 3. Create test dataset
# ----------------------------
test_dataset, scaler, max_features = create_simple_dataset(
    file_ext=data_cfg['test_loop_order'],
    selected_features=features_cfg['selected_features'],
    normalize=True,
    data_dir=data_cfg['base_dir']
)

num_features = test_dataset[0].x.shape[1]  # determine input features from dataset

# Now we can create the model
model = create_gnn_model(
    architecture=model_cfg['name'],
    num_features=num_features,
    hidden_dim=hidden_dim,
    num_classes=1,  # binary classification
    dropout=dropout,
    num_layers=num_layers
).to(device)

# ----------------------------
# 4. Load model checkpoint
# ---------------------------
model_name = "best_model.pt"
model_path = Path(experiment_cfg['model_dir']) / model_name

if not model_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {model_path}")

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# ----------------------------
# 5. Run inference
# ----------------------------
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = torch.sigmoid(out).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

# ----------------------------
# 6. Save predictions
# ----------------------------
output_file = f"test_predictions_loop_{data_cfg['test_loop_order']}.csv"
import numpy as np

df = pd.DataFrame({
    "y_true": all_labels,
    "y_pred": np.array(all_preds).flatten()
})
df.to_csv(output_file, index=False)
print(f"Saved predictions to {output_file}")
