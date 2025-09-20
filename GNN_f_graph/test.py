from torch_geometric.loader import DataLoader
from f_graph_builder import create_simple_dataset_rgcn

# Create DataLoader for your real graphs
loader, scaler, feat_dim = create_simple_dataset_rgcn(
    file_ext='7',                # or whichever loop order / file you need
    selected_features=['degree', 'betweenness', 'clustering'],  # your features
    normalize=True,
    batch_size=4
)

print(f"Feature dimension: {feat_dim}")
print(f"Number of batches: {len(loader)}")
