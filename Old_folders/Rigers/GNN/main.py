import torch
import torch.nn.functional as F
import pandas as pd
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.model_selection import train_test_split
import ast

# ----------------------------
#  Step 1: Load and parse CSVs
# ----------------------------

def parse_edge_line(line):
    # Split by comma only *outside* curly braces
    edges_str = line.strip().split('","')  # Handles entries like "{1, 2}","{1, 3}",...
    cleaned = [edge.replace('{', '(').replace('}', ')').replace('"', '') for edge in edges_str]
    return [tuple(ast.literal_eval(edge)) for edge in cleaned]

# Load the file manually
edges_per_graph = []
with open("/home/rigers/Documents/GitHub/Human-Learning/GNN/data_8_Loop/edges8Loop.csv", 'r') as file:
    for line in file:
        edges = parse_edge_line(line)
        edges_per_graph.append(edges)

labels = []
with open("/home/rigers/Documents/GitHub/Human-Learning/GNN/data_8_Loop/coeffs8Loop.csv", 'r') as f:
    for line in f:
        labels.append(int(line.strip()))

# ----------------------------
#  Step 2: Convert to PyG Data objects
# ----------------------------

graph_list = []
for edges, label in zip(edges_per_graph, labels):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = edge_index.max().item() + 1
    x = torch.eye(num_nodes)  # Identity features
    y = torch.tensor([label], dtype=torch.long)
    graph_list.append(Data(x=x, edge_index=edge_index, y=y))

# ----------------------------
# Step 3: Train/test split
# ----------------------------

train_data, test_data = train_test_split(graph_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# ----------------------------
# Step 4: Define GIN model
# ----------------------------

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super().__init__()
        nn1 = Sequential(Linear(num_node_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)

        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)

        self.linear = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return self.linear(x)

# ----------------------------
#  Step 5: Train & Evaluate
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GIN(num_node_features=graph_list[0].x.size(1), hidden_dim=64, num_classes=len(set(labels))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# --- Training loop
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch:03d}] Loss: {total_loss:.4f}")

# --- Evaluation loop
model.eval()
correct = 0
total = 0
for data in test_loader:
    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
    total += data.num_graphs

accuracy = correct / total
print(pred)
