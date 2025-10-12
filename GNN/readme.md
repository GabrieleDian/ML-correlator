# 🧠 GNN Feature Extraction and Training Pipeline

This repository provides a **complete pipeline** for computing graph-based features, saving them, and training **Graph Neural Networks (GNNs)** on pre-computed features.  
It is designed for research on **planar and f-graph structures**  and supports flexible feature selection and reproducible experiments with Weights & Biases (W&B).

---

## 📁 Repository Structure

```
.
├── compute_features.py       # Compute and save per-graph node features (degree, eigenvectors, etc.)
├── load_features.py          # Load pre-computed features from disk
├── check_features.py         # Inspect, verify, and summarize feature files
├── graph_builder.py          # Build PyTorch Geometric datasets from features
├── GNN_architectures.py      # Collection of GNN models (GIN, GAT, PlanarGNN, etc.)
├── training_utils.py         # Train/evaluate GNNs with metrics and W&B logging
├── one_run_simple.py         # Main script to run one training experiment
├── model_output.py           # Load trained model and generate predictions
├── restore_wandbs.py         # Restore old W&B runs or sweeps
└── Graph_Edge_Data/          # Data directory containing input CSVs and saved features
```

---

## ⚙️ 1. Computing Graph Features

All node-level features are computed using **NetworkX**, **NumPy**, and **SciPy**, and stored as `.npy` files inside  
`Graph_Edge_Data/features_loop_<loop_order>/`.

### Example

```bash
python compute_features.py --file_ext 7 --feature degree
```

This computes the node degrees for all graphs stored in  
`Graph_Edge_Data/den_graph_data_7.csv`.

### Supported features

| Category | Feature name | Description |
|-----------|---------------|--------------|
| Spectral | `eigen_1`–`eigen_3` | Top-3 Laplacian eigenvectors (largest) |
| Spectral | `low_eigen_1`–`low_eigen_3` | Lowest non-zero Laplacian eigenvectors (smoothest modes) |
| Structural | `degree`, `closeness`, `betweenness`, `clustering`, `pagerank` | Classical centrality measures |
| Planar | `face_count` | Counts of faces per node (for planar graphs) |
| Graphlet | `graphlet_2` | Node participation in 2-node graphlets (via GEOMINE) |
| Motif | `W5_indicator` | Detects subgraphs isomorphic to a 5-wheel |
| Matrix-based | `identity_columns`, `adjacency_columns` | Identity / adjacency columns per node |

Features are automatically padded to the largest node count across graphs.

---

## ⚙️ 2. Checking Features

To verify that features were computed correctly:

```bash
python check_features.py --check-all
```

You can also view stats for a single feature:

```bash
python check_features.py --loop 7 --feature degree --stats
```

---

## 🧬 3. Building Datasets

The file `graph_builder.py` combines precomputed features and graph structures into **PyTorch Geometric** datasets.

Example usage inside Python:

```python
from graph_builder import create_simple_dataset
dataset, scaler, feat_dim = create_simple_dataset(
    file_ext='7',
    selected_features=['low_eigen_1', 'degree', 'clustering'],
    normalize=True
)
```

This returns a list of `torch_geometric.data.Data` objects, one per graph.

---

## 🧠 4. Training GNNs

Use `one_run_simple.py` to train a model end-to-end:

```bash
python one_run_simple.py --config configs/config.yaml --train_loop 7 --test_loop 8 --features degree low_eigen_1 low_eigen_2
```

The training script:
- loads the datasets via `graph_builder.py`
- builds a GNN model defined in `GNN_architectures.py`
- trains it using functions from `training_utils.py`
- optionally logs results to **Weights & Biases (wandb)**

### Supported architectures

| Name | Description |
|------|--------------|
| `gin` | Graph Isomorphism Network (recommended baseline) |
| `gat` | Graph Attention Network |
| `hybrid` | Combination of GCN, GIN, GAT layers |
| `planar` | Planar-structure aware GNN |
| `simple` | Lightweight 2-layer GIN (fast baseline) |

---

## 📊 5. Evaluating Models

After training, you can run inference on a dataset using `model_output.py`:

```bash
python model_output.py
```

This script loads the saved model checkpoint and outputs a CSV with  
`y_true` and `y_pred` values for each graph.

---

## 🧹 6. Restoring W&B Runs

To recover old runs or sweeps from local storage:

```bash
python restore_wandbs.py
```

Edit the constants at the top of that file to match your local folder, project, and entity.

---

## 🧪 7. Feature Dictionary Design

All feature computation functions are collected in a single dictionary:

```python
FEATURE_FUNCTIONS = {
    **eigenvector_functions,        # Top eigenvectors
    **lowest_eigenvector_functions, # Bottom eigenvectors
    'degree': compute_degree_features,
    ...
}
```

The `**` operator **unpacks** entire groups of features, allowing modular extension.

---

## 🧠 Notes on Laplacian Eigenvectors

- The **lowest eigenvalue (0)** corresponds to the constant eigenvector.  
- The **multiplicity of 0** equals the number of connected components.  
- The **smallest non-zero eigenvectors** (Fiedler vectors) capture global structure and are used for spectral clustering and smooth graph embeddings.

---

## 🧮 Requirements

```bash
pip install numpy scipy networkx torch torch-geometric scikit-learn tqdm joblib wandb
```

Optionally:
```bash
pip install GEOMINE
```


---

## 🦯 Summary

| Step | Script | Purpose |
|------|---------|----------|
| 1️⃣ | `compute_features.py` | Compute and save node-level features |
| 2️⃣ | `check_features.py` | Inspect features and check consistency |
| 3️⃣ | `graph_builder.py` | Build PyTorch Geometric datasets |
| 4️⃣ | `GNN_architectures.py` | Define model architectures |
| 5️⃣ | `training_utils.py` | Train, evaluate, and log models |
| 6️⃣ | `one_run_simple.py` | Run one end-to-end training |
| 7️⃣ | `model_output.py` | Generate predictions |
| 8️⃣ | `restore_wandbs.py` | Restore old wandb runs |

---


