from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from model.Unet import UNet


def q2(train_data, test_data):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific number of diffusion timesteps. Do this for 10 evenly logarithmically spaced integers
      1 to 512, i.e. np.power(2, np.linspace(0, 9, 10)).astype(int)
    """
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    train_data = 2 * train_data - 1
    test_data = 2 * test_data - 1
    
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 初始化模型
    model = UNet(3, [64, 128, 256, 512], 2).to(device)
    
    try:
        model.load_state_dict(torch.load(r'model\Unet_model.pth'))
        print("模型权重加载成功！")
    except FileNotFoundError:
        print("未找到权重文件，请检查文件路径。")
    except RuntimeError as e:
        print(f"加载权重时出现错误: {e}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    total_steps = 60 * len(test_loader)
    warmup_steps = 100

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    num_epochs = 0
    train_losses = []
    test_losses = []
    best_loss = float('inf')  # 初始化最佳损失为正无穷
    log_file_path = "q2_log.txt"  # 定义日志文件路径

    # test epoch 0
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        for batch in test_loader:
            x = batch[0].to(device)
            t = torch.rand(x.shape[0], 1, 1, 1).to(device)

            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            eps = torch.randn_like(x).to(device)
            x_t = alpha_t * x + sigma_t * eps

            eps_hat = model(x_t, t)
            loss = F.mse_loss(eps_hat, eps)
            total_test_loss += loss.item()
            
        test_losses.append(total_test_loss / len(test_loader))
        # 写入日志
        log_message = f"Epoch {0}/{num_epochs}, Test Loss: {test_losses[-1]}\n"
        with open(log_file_path, 'a') as file:  
            file.write(log_message)
        print(log_message.strip())
            

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # 生成进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, batch in progress_bar:
            x = batch[0].to(device)
            t = torch.rand(x.shape[0], 1, 1, 1).to(device)

            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            eps = torch.randn_like(x).to(device)
            x_t = alpha_t * x + sigma_t * eps

            eps_hat = model(x_t, t)
            loss = F.mse_loss(eps_hat, eps)
            total_train_loss += loss.item()
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            avg_loss = total_train_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})

        # 关闭进度条
        progress_bar.close()
        

        # 计算每个 epoch 结束后的测试损失
        model.eval()
        # 生成进度条
        test_progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test Epoch {epoch + 1}')
        with torch.no_grad():
            total_test_loss = 0
            for batch_idx, batch in test_progress_bar:
                x = batch[0].to(device)
                t = torch.rand(x.shape[0], 1, 1, 1).to(device)

                alpha_t = torch.cos(t * np.pi / 2)
                sigma_t = torch.sin(t * np.pi / 2)
                eps = torch.randn_like(x).to(device)
                x_t = alpha_t * x + sigma_t * eps

                eps_hat = model(x_t, t)
                loss = F.mse_loss(eps_hat, eps)
                total_test_loss += loss.item()
                
            # 关闭进度条
            test_progress_bar.close()    
            test_losses.append(total_test_loss / len(test_loader))

            # 写入日志
            log_message = f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_train_loss / len(train_loader)}, Test Loss: {test_losses[-1]}\n"
            with open(log_file_path, 'a') as file:  
                file.write(log_message)
            print(log_message.strip())
            
            # 保存最佳模型
            if test_losses[-1] < best_loss:
                best_loss = test_losses[-1]
                torch.save(model.state_dict(), 'Unet_model.pth')
                print(f"save best model in epoch {epoch + 1}")


    # 采样
    all_samples = []
    num_steps_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    for num_steps in num_steps_list:
        x = torch.randn(10, 3, 32, 32).to(device)
        ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)
 
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((10, 1, 1, 1), ts[i], dtype=torch.float32).to(device)
                tm1 = torch.full((10, 1, 1, 1), ts[i + 1], dtype=torch.float32).to(device)

                alpha_t = torch.cos(t * np.pi / 2)
                alpha_tm1 = torch.cos(tm1 * np.pi / 2)
                sigma_t = torch.sin(t * np.pi / 2)
                sigma_tm1 = torch.sin(tm1 * np.pi / 2)

                eps_hat = model(x, t)
    
                # 裁剪
                x_hat = torch.clamp((x - sigma_t * eps_hat) / alpha_t, -1, 1)
                
                eta_t = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t ** 2 / alpha_tm1 ** 2)
                t1 = alpha_tm1 * x_hat
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

    all_samples = np.array(all_samples)
    all_samples = (all_samples + 1) / 2
    all_samples = np.transpose(all_samples, (0, 1, 3, 4, 2))

    return np.array(train_losses), np.array(test_losses), all_samples

q2_save_results(q2)