import os

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SimpleDataset(Dataset):
    def __init__(self, dim_x, dim_y, data_amount):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.data_amount = data_amount

    def __len__(self):
        return self.data_amount

    def __getitem__(self, item):
        return {'x': torch.rand(size=(self.dim_x,)),
                'y': torch.rand(size=(self.dim_y,)),
                'name': f'data {item}'}


class SimpleNet(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.net(x)


def train():
    """
    包含了数据集、神经网络、损失函数、optimizer、scheduler的自定义，包含了训练和验证，包含了训练过程的记录（基于wandb）和模型的保存。
    """
    batch_size = 64
    learning_rate = 1e-3
    max_epoch = 20
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_set = SimpleDataset(dim_x=10, dim_y=3, data_amount=10000)
    valid_set = SimpleDataset(dim_x=10, dim_y=3, data_amount=1000)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    loss_func = torch.nn.MSELoss()
    network = SimpleNet(dim_in=10, dim_out=3).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=128, eta_min=2e-5)

    wandb.init(project='TrainSimpleNet', name='sample')

    for epoch in range(1, max_epoch + 1):
        with tqdm(desc=f'epoch {epoch}/{max_epoch}', total=len(train_loader) * batch_size, unit='items') as pbar:
            network.train()
            for train_batch in train_loader:
                inputs = train_batch['x'].to(device)
                truth = train_batch['y'].to(device)
                out = network(inputs)
                loss = loss_func(out, truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                wandb.log({'tr_loss': loss.item(),
                           'epoch': epoch,
                           'lr': optimizer.param_groups[0]['lr']})
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
                pbar.update(batch_size)
            torch.save(network.state_dict(), os.path.join(f'epoch_{epoch}.pth'))
        valid_loss = []
        with torch.no_grad():
            network.eval()
            for valid_data in valid_loader:
                inputs = valid_data['x'].to(device)
                truth = valid_data['y'].to(device)
                out = network(inputs)
                loss = loss_func(out, truth)
                valid_loss.append(loss.item())
            print(f'valid after epoch {epoch}, loss {np.mean(valid_loss):.3f}')
        wandb.log({'va_loss': np.mean(valid_loss)})


if __name__ == '__main__':
    train()
