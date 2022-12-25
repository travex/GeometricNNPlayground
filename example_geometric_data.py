import torch
from torch_geometric.data import Data


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

data.validate(raise_on_error=True)

print(data.keys)

print(data['edge_index'])

print("Is graph undirected?",data.is_undirected())