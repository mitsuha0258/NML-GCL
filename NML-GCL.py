import torch
import GCL.augmentors as A
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import numpy as np

from utils import seed_everything, load_dataset, cos_sim
from layers import GConv, GRACE, MLP
from eval import test


class Encoder(GRACE):
    def __init__(self, encoder, augmentor, output_dim, proj_dim, tau, use_mlp=False):
        super(Encoder, self).__init__(encoder, augmentor, output_dim, proj_dim, tau)

        self.hard_mask = None
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(output_dim, emb_dim),
            activation(),
            torch.nn.Linear(emb_dim, output_dim),
        )
        self.weight_init()
        self.freeze_params(flag=False)
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
    
    def freeze_params(self, flag=True):
        if flag:
            for name, param in self.named_parameters():
                if 'mlp1' not in name:
                    param.requires_grad = False
                    # print(name, param)
                else:
                    param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if 'mlp1' in name:
                    param.requires_grad = False
                    # print(name, param)
                else:
                    param.requires_grad = True

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        
        neg = f(cos_sim(z1, z2))
        pos = neg.diag()
        
        weight_mt = cos_sim(self.mlp1(z1), self.mlp1(z2))
        weight_mt = torch.nn.functional.softmax(weight_mt, dim=1)
        
        uniform_mt = torch.ones_like(weight_mt, device=z1.device) / weight_mt.shape[0]
        l1 = torch.nn.functional.kl_div(torch.log(weight_mt), uniform_mt, reduction='batchmean')
        
        weight_mt = weight_mt * (weight_mt.shape[0] - 1)
        loss = -torch.log(
                pos / ( (neg * weight_mt).sum(1) + pos  )
            )
        return loss + alpha * l1


def train(encoder_model, data, optimizer, epoch):
    encoder_model.train()
    
    aug1, aug2 = encoder_model.augmentor
    x1, edge_index1, edge_weight1 = aug1(data.x, data.edge_index)
    x2, edge_index2, edge_weight2 = aug2(data.x, data.edge_index)
    
    total_loss = []
    # min M
    encoder_model.freeze_params(flag=True)
    for _ in range(epoch_m):
        optimizer.zero_grad()
        z1 = encoder_model.encoder(x1, edge_index1, edge_weight1)
        z2 = encoder_model.encoder(x2, edge_index2, edge_weight2)
        loss = encoder_model.loss(z1, z2, mean=True)
        # total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    # min E
    encoder_model.freeze_params(flag=False)
    for _ in range(epoch_e):
        optimizer.zero_grad()
        z1 = encoder_model.encoder(x1, edge_index1, edge_weight1)
        z2 = encoder_model.encoder(x2, edge_index2, edge_weight2)
        loss = encoder_model.loss(z1, z2, mean=True)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(total_loss)


def main():

    aug1 = A.Compose([A.EdgeRemoving(pe=pe1), A.FeatureMasking(pf=pf1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=pe2), A.FeatureMasking(pf=pf2)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=hidden_dim, output_dim=output_dim, activation=activation, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), output_dim=output_dim, proj_dim=proj_dim, tau=tau).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr1, weight_decay=wd1)

    with tqdm(total=num_epochs, desc='(T)', ncols=100) as pbar:
        for epoch in range(1, num_epochs+1):
            
            # 训练
            loss = train(encoder_model, data, optimizer, epoch)
            
            
            pbar.set_postfix({'loss': loss})
            pbar.update()

    acc, std = test(encoder_model, data, dataset_name, repeat_times=repeat_times, test_epochs=test_epochs, lr2=lr2, test_interval=2, wd2=wd2)
    print(f'{dataset_name} Average accuracy: {acc:.2f} + {std:.2f}')
    logging.info(f'{dataset_name} Average accuracy: {acc:.4f} +- {std:.4f}')
        

if __name__ == '__main__':
    import logging
    log_name = 'results'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filename=f'./{log_name}.log'
    )
    
    method_name = 'NML-GCL'
    logging.info(f"============= {method_name} ==============")
    
    device = torch.device('cuda')
    seed = 15
    
    
    repeat_times = 10
    
    test_epochs = 3000
    lr2 = 1e-2
    wd2 = 0
    tau2 = 0.1
    
    batch_size = 0
    pf1, pf2, pe1, pe2 = 0.1, 0.1, 0.4, 0.4
    
    hidden_dim = 512 
    output_dim, proj_dim = 512, 0
    emb_dim = output_dim
    activation = torch.nn.RReLU

    path = './datasets'
    dataset_name = 'Cora' # 'Cora', 'CiteSeer', 'PubMed'
    dataset = load_dataset(path, dataset_name)
    data = dataset[0].to(device)
    
    alpha = 0
    if dataset_name == 'Cora':
        pf1, pf2, pe1, pe2 = 0.1, 0.1, 0.4, 0.4
        lr1, wd1 = 5e-4, 1e-3
        tau = 0.8
        num_epochs = 200
        epoch_m = 2
        epoch_e = 1
        lr2 = 1e-3
        alpha = (data.num_nodes - 1) * 0.1
        
    elif dataset_name == 'CiteSeer':
        pf1, pf2, pe1, pe2 = 0.1, 0.1, 0.4, 0.4
        lr1, wd1 = 5e-4, 5e-3
        tau = 0.7
        num_epochs = 50
        epoch_m = 3
        epoch_e = 1
        lr2 = 1e-3
        alpha = (data.num_nodes - 1) * 0.1
    elif dataset_name == 'PubMed':
        pf1, pf1, pe1, pe2 = 0.1, 0.1, 0.4, 0.4
        lr1, wd1 = 5e-4, 0
        hidden_dim, proj_dim = 512, 0
        tau = 0.5
        num_epochs = 150
        epoch_m = 3
        epoch_e = 1
        lr2 = 1e-3
        alpha = (data.num_nodes - 1) * 0.05
    elif dataset_name == 'Computers':
        pf1, pf1, pe1, pe2 = 0.1, 0.1, 0.4, 0.4
        lr1, wd1 = 5e-4, 0
        hidden_dim, proj_dim = 512, 0
        tau = 0.4
        num_epochs = 50
        epoch_m = 8
        epoch_e = 1
        lr2 = 1e-2
        alpha = (data.num_nodes - 1) * 0.1
    elif dataset_name == 'Photo':
        pf1, pf2, pe1, pe2 = 0.1, 0.1, 0.4, 0.4
        lr1, wd1 = 1e-4, 0
        hidden_dim, proj_dim = 512, 0
        tau = 0.5
        num_epochs = 50
        epoch_m = 5
        epoch_e = 1
        lr2 = 1e-2
        alpha = (data.num_nodes - 1) * 0.10
    elif dataset_name == 'WikiCS':
        pf1, pf2, pe1, pe2 = 0.1, 0.1, 0.2, 0.2
        lr1, wd1 = 5e-4, 0
        hidden_dim, proj_dim = 512, 0
        tau = 0.5
        num_epochs = 50
        epoch_m = 6
        epoch_e = 1
        lr2 = 1e-2
        alpha = (data.num_nodes - 1) * 0.2
    
    seed_everything(seed=seed)
    main()
    
    logging.info("")
    


