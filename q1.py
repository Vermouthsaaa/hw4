from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers):
        super(MLP, self).__init__()
        layers = []

        for i in range(num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def q1(train_data, test_data):
    """
    train_data: A (100000, 2) numpy array of 2D points
    test_data: A (10000, 2) numpy array of 2D points

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (9, 2000, 2) of samples drawn from your model.
      Draw 2000 samples for each of 9 different number of diffusion sampling steps
      of evenly logarithmically spaced integers 1 to 512
      hint: np.power(2, np.linspace(0, 9, 9)).astype(int)
    """
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # 初始化模型
    model = MLP(input_size=3, hidden_size=64, num_hidden_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    total_steps = 100 * len(train_loader)
    warmup_steps = 100

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_epochs = 100
    train_losses = []
    test_losses = []

    # test epoch 0
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        for batch in test_loader:
            x = batch[0].to(device)
            t = torch.rand(x.shape[0], 1).to(device)

            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            eps = torch.randn_like(x).to(device)
            x_t = alpha_t * x + sigma_t * eps

            input_x = torch.cat([x_t, t], dim=1)
            eps_hat = model(input_x)
            loss = F.mse_loss(eps_hat, eps)
            total_test_loss += loss.item()
            
        test_losses.append(total_test_loss / len(test_loader))
        print(f"Epoch {0}/{num_epochs}, Test Loss: {test_losses[-1]}")

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            t = torch.rand(x.shape[0], 1).to(device)

            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            eps = torch.randn_like(x).to(device)
            x_t = alpha_t * x + sigma_t * eps

            input_x = torch.cat([x_t, t], dim=1)
            eps_hat = model(input_x)
            loss = F.mse_loss(eps_hat, eps)
            total_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        # 计算每个 epoch 结束后的测试损失
        model.eval()
        with torch.no_grad():
            total_test_loss = 0
            for batch in test_loader:
                x = batch[0].to(device)
                t = torch.rand(x.shape[0], 1).to(device)

                alpha_t = torch.cos(t * np.pi / 2)
                sigma_t = torch.sin(t * np.pi / 2)
                eps = torch.randn_like(x).to(device)
                x_t = alpha_t * x + sigma_t * eps

                input_x = torch.cat([x_t, t], dim=1)
                eps_hat = model(input_x)
                loss = F.mse_loss(eps_hat, eps)
                total_test_loss += loss.item()
                
            test_losses.append(total_test_loss / len(test_loader))
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_train_loss / len(train_loader)}, Test Loss: {test_losses[-1]}")

    # 采样
    all_samples = []
    num_steps_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for num_steps in num_steps_list:
        x = torch.randn(2000, 2).to(device)
        ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)

        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((2000, 1), ts[i], dtype=torch.float32).to(device)
                tm1 = torch.full((2000, 1), ts[i + 1], dtype=torch.float32).to(device)

                alpha_t = torch.cos(t * np.pi / 2)
                alpha_tm1 = torch.cos(tm1 * np.pi / 2)
                sigma_t = torch.sin(t * np.pi / 2)
                sigma_tm1 = torch.sin(tm1 * np.pi / 2)

                input_x = torch.cat([x, t], dim=1)
                eps_hat = model(input_x)
                eta_t = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t ** 2 / alpha_tm1 ** 2)
                t1 = alpha_tm1 * ((x - sigma_t * eps_hat) / alpha_t)
                t2 = torch.sqrt(torch.clamp(sigma_tm1 ** 2 - eta_t ** 2, min=0)) * eps_hat
                # if i < num_steps - 1:
                #     eps_t = torch.randn_like(x)
                #     t3 = eta_t * eps_t
                # else:
                #     t3 = torch.zeros_like(x)
                eps_t = torch.randn_like(x)
                t3 = eta_t * eps_t
                x = t1 + t2 + t3

            all_samples.append(x.cpu().numpy())

    return np.array(train_losses), np.array(test_losses), np.array(all_samples)
    
q1_save_results(q1)