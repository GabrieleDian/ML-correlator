import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

class GraphBuilder:
    def __init__(self, solid_edges, dashed_edges, node_labels=None):
        # Auto-infer node labels if not provided
        if node_labels is None:
            node_labels = sorted(set(u for e in solid_edges + dashed_edges for u in e))
        self.node_labels = node_labels
        self.label2idx = {label: i for i, label in enumerate(node_labels)}

        self.solid_edges = solid_edges
        self.dashed_edges = dashed_edges
        self.num_nodes = len(self.node_labels)

    def build(self, extra_node_features=None):
        edge_list = []
        edge_types = []

        for u, v in self.solid_edges:
            i, j = self.label2idx[u], self.label2idx[v]
            edge_list += [[i, j], [j, i]]  # bidirectional
            edge_types += [0, 0]

        for u, v in self.dashed_edges:
            i, j = self.label2idx[u], self.label2idx[v]
            edge_list += [[i, j], [j, i]]
            edge_types += [1, 1]

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # Basic node feature: degree
        degree_feat = degree(edge_index[0], num_nodes=self.num_nodes).view(-1, 1)

        # Combine degree with extra features if provided
        if extra_node_features is not None:
            assert extra_node_features.shape[0] == self.num_nodes, \
                "extra_node_features must match number of nodes"
            x = torch.cat([degree_feat, extra_node_features], dim=1)
        else:
            x = degree_feat

        return Data(x=x, edge_index=edge_index, edge_type=edge_type, num_nodes=self.num_nodes)


if __name__ == "__main__":
    # --- Example for your 13-node graph (9-loop) ---
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
    graph = builder.build()

    print(graph)
