from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from model.DIT import DiT

    
def q3_b(train_data, train_labels, test_data, test_labels, vae):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    train_labels: A (50000,) numpy array of class labels
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]
    test_labels: A (10000,) numpy array of class labels
    vae: a pretrained VAE

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific class (i.e. row 0 is class 0, row 1 class 1, ...). Use 512 diffusion timesteps
    """

    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    train_data = 2 * train_data - 1
    test_data = 2 * test_data - 1
    
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 初始化模型
    model = DiT(input_shape=(4, 8, 8), patch_size=2, hidden_size=512, num_heads=8, num_layers=12, num_classes=10, cfg_dropout_prob=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    total_steps = 60 * len(train_loader)
    warmup_steps = 100

    try:
        model.load_state_dict(torch.load(r'model\Dit2_model.pth'))
        print("模型权重加载成功！")
    except FileNotFoundError:
        print("未找到权重文件，请检查文件路径。")
    except RuntimeError as e:
        print(f"加载权重时出现错误: {e}")
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_epochs = 0
    train_losses = []
    test_losses = []
    best_loss = float('inf')  # 初始化最佳损失为正无穷
    log_file_path = "q3_b_2_log.txt"  # 定义日志文件路径

    # test epoch 0
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            # 编码
            x = vae.encode(x)
            # scale_factor = 1.046503
            scale_factor = 1.04
            x = x / scale_factor

            t = torch.rand(x.shape[0], 1).to(device)
            alpha_t = torch.cos(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
            sigma_t = torch.sin(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
            eps = torch.randn_like(x).to(device)
            x_t = alpha_t * x + sigma_t * eps

            eps_hat = model(x_t, y, t)
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
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, batch in progress_bar:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            # 编码
            x = vae.encode(x)
            # scale_factor = 1.046503
            scale_factor = 1.04
            x = x / scale_factor

            t = torch.rand(x.shape[0], 1).to(device)
            alpha_t = torch.cos(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
            sigma_t = torch.sin(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
            eps = torch.randn_like(x).to(device)
            x_t = alpha_t * x + sigma_t * eps

            eps_hat = model(x_t, y, t)
            loss = F.mse_loss(eps_hat, eps)
            total_train_loss += loss.item()
            train_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            avg_loss = total_train_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})

        # 关闭进度条
        progress_bar.close()

        # 计算每个 epoch 结束后的测试损失
        model.eval()
        test_progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test Epoch {epoch + 1}')
        with torch.no_grad():
            total_test_loss = 0
            for batch_idx, batch in test_progress_bar:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                # 编码
                x = vae.encode(x)
                # scale_factor = 1.046503
                scale_factor = 1.04
                x = x / scale_factor

                t = torch.rand(x.shape[0], 1).to(device)
                alpha_t = torch.cos(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
                sigma_t = torch.sin(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
                eps = torch.randn_like(x).to(device)
                x_t = alpha_t * x + sigma_t * eps

                eps_hat = model(x_t, y, t)
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
            torch.save(model.state_dict(), 'Dit2_model.pth')
            print(f"save best model in epoch {epoch + 1}")
        

    # 采样
    print("start samples:")
    samples = []
    num_steps = 512
    for class_idx in range(10):
        print(f"generate class {class_idx}")

        x = torch.randn(10, 4, 8, 8).to(device)
        y = torch.tensor([class_idx], dtype=torch.long).repeat(10).to(device)
        ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)

        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((10, 1), ts[i], dtype=torch.float32).to(device)
                tm1 = torch.full((10, 1), ts[i + 1], dtype=torch.float32).to(device)

                alpha_t = torch.cos(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
                alpha_tm1 = torch.cos(tm1 * np.pi / 2).unsqueeze(2).unsqueeze(3)
                sigma_t = torch.sin(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
                sigma_tm1 = torch.sin(tm1 * np.pi / 2).unsqueeze(2).unsqueeze(3)

                eps_hat = model(x, y, t)
                
                # 裁剪
                # x_hat = torch.clamp((x - sigma_t * eps_hat) / alpha_t, -1, 1)
                
                eta_t = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t ** 2 / alpha_tm1 ** 2)
                # t1 = alpha_tm1 * x_hat
                t1 = alpha_tm1 * ((x - sigma_t * eps_hat) / alpha_t)
                t2 = torch.sqrt(torch.clamp(sigma_tm1 ** 2 - eta_t ** 2, min=0)) * eps_hat
                eps_t = torch.randn_like(x)
                t3 = eta_t * eps_t
                x = t1 + t2 + t3

        # scale_factor = np.std(x.detach().cpu().numpy().flatten())
        # print(scale_factor)
        # scale_factor = 1.046503
        # 解码
        scale_factor = 1.04
        x = x * scale_factor
        x = vae.decode(x)
        samples.append(x.cpu().numpy())
        
    
    samples = np.array(samples)
    samples = np.transpose(samples, (0, 1, 3, 4, 2))
    samples = (samples + 1) / 2

    return np.array(train_losses), np.array(test_losses), samples

q3b_save_results(q3_b)