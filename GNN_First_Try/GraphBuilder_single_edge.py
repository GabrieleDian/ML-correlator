import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


class GraphBuilder:
    def __init__(self, solid_edges, coeff, node_labels=None):
        # Auto-infer node labels if not provided
        if node_labels is None:
            node_labels = sorted(set(u for e in solid_edges for u in e))
        self.node_labels = node_labels
        self.label2idx = {label: i for i, label in enumerate(node_labels)}

        self.solid_edges = solid_edges
        self.num_nodes = len(self.node_labels)
        self.y = torch.tensor(coeff, dtype=torch.long)  # Ensure coeff is a column vector

    def build(self, extra_node_features=None):
        edge_list = []

        for u, v in self.solid_edges:
            i, j = self.label2idx[u], self.label2idx[v]
            edge_list += [[i, j], [j, i]]  # bidirectional

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Basic node feature: degree
        degree_feat = degree(edge_index[0], num_nodes=self.num_nodes).view(-1, 1)

        # Combine degree with extra features if provided
        if extra_node_features is not None:
            assert extra_node_features.shape[0] == self.num_nodes, \
                "extra_node_features must match number of nodes"
            x = torch.cat([degree_feat, extra_node_features], dim=1)
        else:
            x = degree_feat
        return Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes, coeff=self.y)
