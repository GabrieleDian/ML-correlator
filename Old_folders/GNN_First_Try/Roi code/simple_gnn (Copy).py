import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import Data
from graph_builder import GraphBuilder  # <-- External builder

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations=2):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        batch = torch.zeros(data.num_nodes, dtype=torch.long)  # single graph => single batch
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)  # mean over all nodes
        return self.lin(x)

if __name__ == "__main__":
    # --- 13-node example ---
    solid_edges = [
        ('a', 'b'), ('a', 'k'), ('a', 'e'), ('a', 'j'), ('a', 'f'), ('a', 'l'),
        ('b', 'l'), ('b', 'c'), ('b', 'h'), ('b', 'i'), ('b', 'k'), ('c', 'l'),
        ('c', 'f'), ('c', 'g'), ('c', 'm'), ('c', 'h'), ('d', 'e'), ('d', 'i'),
        ('d', 'h'), ('d', 'm'), ('d', 'g'), ('d', 'j'), ('e', 'k'), ('e', 'i'),
        ('e', 'j'), ('f', 'j'), ('f', 'g'), ('f', 'l'), ('g', 'j'), ('g', 'm'),
        ('h', 'm'), ('h', 'i'), ('i', 'k')
    ]
    dashed_edges = [
        ('a', 'c'), ('a', 'i'), ('b', 'g'), ('b', 'j'), ('c', 'd'),
        ('d', 'f'), ('e', 'h')
    ]

    builder = GraphBuilder(solid_edges, dashed_edges)
    data = builder.build()  # Returns a torch_geometric.data.Data object

    # --- Initialize GNN ---
    in_channels = data.x.shape[1]
    model = SimpleGNN(in_channels=in_channels, hidden_channels=16)

    # --- Run forward pass ---
    model.eval()
    with torch.no_grad():
        output = model(data)

    print("GNN output (untrained):", output.item())
