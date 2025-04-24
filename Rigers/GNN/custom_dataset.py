import torch
import pandas as pd
import ast
from torch_geometric.data import Data, InMemoryDataset

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, edge_path, label_path, transform=None):
        self.edge_path = edge_path
        self.label_path = label_path
        super().__init__('.', transform)
        self.data, self.slices = self.process()

    def process(self):
        edge_df = pd.read_csv(self.edge_path, header=None)
        label_df = pd.read_csv(self.label_path, header=None)

        data_list = []

        for idx, row in edge_df.iterrows():
            edge_list = ast.literal_eval(row[0])  # safely parse string -> list of tuples
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # Dummy node features: one-hot vector per node
            num_nodes = edge_index.max().item() + 1
            x = torch.eye(num_nodes)

            y = torch.tensor([label_df.iloc[idx, 0]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        return self.collate(data_list)
