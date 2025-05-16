import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


from utils import cos_sim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, num_layers):
        super(MLP, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x, edge_index, edge_weight):
        z = x
        for i, layer in enumerate(self.layers[:-1]):
            z = layer(z)
            z = self.activation(z)
        z = self.layers[-1](z)
        return z


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(1, num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers[:-1]):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        z = self.layers[-1](z, edge_index, edge_weight)
        return z
    
    
class GRACE(nn.Module):
    def __init__(self, encoder, augmentor, output_dim, proj_dim, tau = 0.5, batch_size = 0):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.tau = tau
        self.proj_dim = proj_dim
        self.batch_size = batch_size

        self.fc1 = torch.nn.Linear(output_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, output_dim)
        
        self.hard_mask = None


    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        z = self.fc2(z)
        return z

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        
        if self.hard_mask is None:
            self.hard_mask = torch.ones(z1.size(0), z2.size(0)).to(z1.device)
        
        pos = f(cos_sim(z1, z2)).diag()
        refl_sim = f(cos_sim(z1, z1)) * self.hard_mask
        between_sim = f(cos_sim(z1, z2)) * self.hard_mask
        
        loss = -torch.log(
                pos / (refl_sim.sum(1) + between_sim.sum(1) + pos - refl_sim.diag() - between_sim.diag())
            )
        # loss = -torch.log(
        #         pos / (between_sim.sum(1) + pos - between_sim.diag())
        #     )
        return loss

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        
        if self.hard_mask is None:
            self.hard_mask = torch.ones(z1.size(0), z2.size(0)).to(z1.device)

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            pos = f(cos_sim(z1[mask], z2))[:, i * batch_size:(i + 1) * batch_size].diag()
            refl_sim = f(cos_sim(z1[mask], z1)) * self.hard_mask[mask]  # [B, N]
            between_sim = f(cos_sim(z1[mask], z2)) * self.hard_mask[mask]  # [B, N]

            losses.append(-torch.log(
                pos / (refl_sim.sum(1) + between_sim.sum(1) + pos
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag() - between_sim[:, i * batch_size:(i + 1) * batch_size].diag())
            ))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        if self.proj_dim != 0:
            h1 = self.project(z1)
            h2 = self.project(z2)
        else:   
            h1, h2 = z1, z2

        if self.batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, self.batch_size)
            l2 = self.batched_semi_loss(h2, h1, self.batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def get_embedding(self, x, edge_index, edge_weight=None):
        with torch.no_grad():
            return self.encoder(x, edge_index, edge_weight)