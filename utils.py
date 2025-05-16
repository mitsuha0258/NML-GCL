import torch
import random
import os
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, WikiCS, Coauthor
from torch_geometric.utils import to_dense_adj, to_edge_index

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import WikiCSDataset

import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def load_dataset(path, name, normalize=False):
    # path = './datasets'
    transform = T.NormalizeFeatures() if normalize else None
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name=name, transform=transform)
       
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name=name, transform=transform)
        
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(path, name=name, transform=transform)
        
    elif name in ['WikiCS']:
        # path1 = './datasets/WikiCS'
        path = path + '/WikiCS'
        dataset = WikiCS(path, transform=transform)
        
    else:
        raise ValueError('Invalid dataset name')
    return dataset


def load_dataset_dgl(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'ama_photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'ama_computer':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'co_physics':
        dataset = CoauthorPhysicsDataset()
    elif name == 'co_cs':
        dataset = CoauthorCSDataset()
    elif name == 'wiki':
        dataset = WikiCSDataset()
    else:
        raise NotImplementedError("Unexpected Dataset")
    
    return dataset


def dropout_edge_by_num(data, type, num=400):
    if num == 0:
        return data.edge_index

    row, col = data.edge_index
    if type == "intro":
        edge_mask = (data.y[row] == data.y[col])
        # print("intro edge num:", edge_mask.sum()/2)
    elif type == "inter":
        edge_mask = (data.y[row] != data.y[col])
        # print("inter edge num:", edge_mask.sum()/2)
    elif type == "all":
        edge_mask = torch.ones(row.size(0), dtype=torch.bool)
        # print("all edge num:", edge_mask.sum()/2)
    
    retain_edge_index = data.edge_index[:, ~edge_mask]
    edge_mask[row > col] = False
    drop_edge_index = data.edge_index[:, edge_mask]
    
    drop_columns = random.sample(range(drop_edge_index.size(1)), drop_edge_index.size(1)-num)
    drop_edge_index = drop_edge_index[:, drop_columns]
    edge_index = torch.cat([retain_edge_index, drop_edge_index, drop_edge_index.flip(0)], dim=1)

    # print("del edge num:", num)
    return edge_index

def add_edge_by_num(my_data, type, num=400):
    if num == 0:
        return my_data.edge_index

    adj = to_dense_adj(my_data.edge_index, max_num_nodes=my_data.num_nodes).squeeze()
    adj_inversed = (1 - adj).bool()

    if type == "intro":
        label_mask = (my_data.y.unsqueeze(0) == my_data.y.unsqueeze(1))
        label_mask = adj_inversed & label_mask
        # print("intro edge num:", (label_mask.sum()-my_data.num_nodes)/2)
    elif type == "inter":
        label_mask = (my_data.y.unsqueeze(0) != my_data.y.unsqueeze(1))
        label_mask = adj_inversed & label_mask
        # print("intro edge num:", label_mask.sum()/2)
    elif type == "all":    
        label_mask = adj_inversed 
        # print("all edge num:", (label_mask.sum()-my_data.num_nodes)/2)
    
    # print(label_mask.shape)
    label_mask.fill_diagonal_(False)
    # Get the indices of k true values in same_label_mask
    x, y = torch.nonzero(label_mask, as_tuple=True)
    mask = x > y
    x, y = x[mask], y[mask]
    false_indices = random.sample(range(x.size(0)), x.size(0)-num)

    # Set the selected indices to false in same_label_mask
    label_mask[x[false_indices], y[false_indices]] = False
    label_mask[y[false_indices], x[false_indices]] = False

    # Update the adjacency matrix adj using the modified same_label_mask
    
    adj = (adj + label_mask.int()).to_sparse()

    return to_edge_index(adj)[0]

def dropout_adj_by_label(edge_index, y, type, p):
    
    if p == 0:
        return edge_index

    edge_num = edge_index.size(1) / 2
    ptb_num = int(edge_num * p)

    row, col = edge_index
    if type == "intro":
        edge_mask = (y[row] == y[col])
        # print("intro edge num:", edge_mask.sum()/2)
    elif type == "inter":
        edge_mask = (y[row] != y[col])
        # print("inter edge num:", edge_mask.sum()/2)
    elif type == "all":
        edge_mask = torch.ones(row.size(0), dtype=torch.bool)
        # print("all edge num:", edge_mask.sum()/2)
    
    retain_edge_index = edge_index[:, ~edge_mask]
    edge_mask[row > col] = False
    drop_edge_index = edge_index[:, edge_mask]
    
    drop_columns = random.sample(range(drop_edge_index.size(1)), drop_edge_index.size(1)-ptb_num)
    drop_edge_index = drop_edge_index[:, drop_columns]
    edge_index = torch.cat([retain_edge_index, drop_edge_index, drop_edge_index.flip(0)], dim=1)

    return edge_index

def add_adj_by_label(edge_index, y, type, p):

    if p == 0:
        return edge_index
    
    num_nodes = y.size(0)
    edge_num = edge_index.size(1) / 2
    ptb_num = int(edge_num * p)

    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze()
    adj_inversed = (1 - adj).bool()

    if type == "intro":
        label_mask = (y.unsqueeze(0) == y.unsqueeze(1))
        label_mask = adj_inversed & label_mask
        # print("intro edge num:", (label_mask.sum()-my_data.num_nodes)/2)
    elif type == "inter":
        label_mask = (y.unsqueeze(0) != y.unsqueeze(1))
        label_mask = adj_inversed & label_mask
        # print("intro edge num:", label_mask.sum()/2)
    elif type == "all":    
        label_mask = adj_inversed 
        # print("all edge num:", (label_mask.sum()-my_data.num_nodes)/2)
    
    # print(label_mask.shape)
    label_mask.fill_diagonal_(False)
    # Get the indices of k true values in same_label_mask
    x, y = torch.nonzero(label_mask, as_tuple=True)
    mask = x > y
    x, y = x[mask], y[mask]
    add_indices = random.sample(range(x.size(0)), ptb_num)

    add_edge_adj = torch.zeros_like(label_mask)
    add_edge_adj[x[add_indices], y[add_indices]] = 1
    add_edge_adj[y[add_indices], x[add_indices]] = 1
    
    adj = (adj + add_edge_adj).to_sparse()

    return to_edge_index(adj)[0]

    
def k_order_path(adj, k=2, normalize=False):
    
    if normalize:
        D = adj.sum(dim=1)
        D_hat_inv_sqrt = torch.diag(D.pow(-0.5))
        adj = D_hat_inv_sqrt @ adj @ D_hat_inv_sqrt
    
    if k == 1:
        return adj
    
    ori_adj = adj
    for _ in range(k-1):
        adj = adj @ ori_adj
        
    return adj


def cos_sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    



if __name__ == "__main__":
    # 生成全连接图的边索引
    # edge_index = torch.LongTensor([[0, 0, 0, 1, 3, 2],
    #                             [1, 3, 2, 0, 0, 0]])

    # num_nodes = 4
    # row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    # col = torch.arange(num_nodes).repeat(num_nodes)
    # full_edge_index = torch.stack([row, col], dim=0)
    # mask = row != col  # 去掉自环
    # full_edge_index = full_edge_index[:, mask]

    # random_bool_matrix = torch.randint(0, 2, (4, 4), dtype=torch.bool)
    # print(random_bool_matrix)
    # print(random_bool_matrix[[0, 0], [0, 1]])
    data = load_dataset('./datasets', 'Cora')[0]
    from torch_geometric.utils import k_hop_subgraph
    print(k_hop_subgraph(0, num_hops=2, edge_index=data.edge_index)[0])
    print(data.y[0] == data.y[1990])
    print(cos_sim(data.x[0].unsqueeze(0), data.x[1990].unsqueeze(0)))
    