from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from model.DIT import DiT


def q3_c(vae):
    """
    vae: a pretrained vae

    Returns
    - a numpy array of size (4, 10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 4 x 10 x 10 grid of generated samples - 4 10 x 10 grid of samples
      with 4 different CFG values of w = {1.0, 3.0, 5.0, 7.5}. Each row of the 10 x 10 grid
      should contain samples of a different class. Use 512 diffusion sampling timesteps.
    """
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化 GPU
    model = DiT(input_shape=(4, 8, 8), patch_size=2, hidden_size=512, num_heads=8, num_layers=12, num_classes=10, cfg_dropout_prob=0.1).to(device)
    
    try:
        model.load_state_dict(torch.load(r'model\Dit2_model.pth'))
        print("model 模型权重加载成功！")
    except FileNotFoundError:
        print("未找到权重文件，请检查文件路径。")
    except RuntimeError as e:
        print(f"加载权重时出现错误: {e}")

    # 存储不同 CFG 值的采样结果
    all_samples = []
    cfg_values = [1.0, 3.0, 5.0, 7.5]
    num_steps = 512

    for cfg in cfg_values:
        class_samples = []
        for class_idx in range(10):
            print(f"cfg: {cfg}, class {class_idx}")

            x = torch.randn(10, 4, 8, 8).to(device)
            y = torch.tensor([class_idx], dtype=torch.long).repeat(10).to(device)
            y_non = torch.tensor([10], dtype=torch.long).repeat(10).to(device)  # 无条件模型的类别标签
            ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)

            with torch.no_grad():
                for i in range(num_steps):
                    t = torch.full((10, 1), ts[i], dtype=torch.float32).to(device)
                    tm1 = torch.full((10, 1), ts[i + 1], dtype=torch.float32).to(device)

                    alpha_t = torch.cos(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
                    alpha_tm1 = torch.cos(tm1 * np.pi / 2).unsqueeze(2).unsqueeze(3)
                    sigma_t = torch.sin(t * np.pi / 2).unsqueeze(2).unsqueeze(3)
                    sigma_tm1 = torch.sin(tm1 * np.pi / 2).unsqueeze(2).unsqueeze(3)

                    # 无条件预测
                    eps_hat_non = model(x, y_non, t)
                    # 有条件预测
                    eps_hat = model(x, y, t)

                    # CFG 公式
                    eps_hat = eps_hat_non + cfg * (eps_hat - eps_hat_non)
                    
                    eta_t = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t ** 2 / alpha_tm1 ** 2)
                    t1 = alpha_tm1 * ((x - sigma_t * eps_hat) / alpha_t)
                    t2 = torch.sqrt(torch.clamp(sigma_tm1 ** 2 - eta_t ** 2, min=0)) * eps_hat
                    eps_t = torch.randn_like(x)
                    t3 = eta_t * eps_t
                    x = t1 + t2 + t3

            scale_factor = 1.04
            x = x * scale_factor
            x = vae.decode(x)
            class_samples.append(x.cpu().numpy())

        class_samples = np.array(class_samples)
        class_samples = np.transpose(class_samples, (0, 1, 3, 4, 2))
        class_samples = (class_samples + 1) / 2
        
        all_samples.append(class_samples)

    all_samples = np.array(all_samples)
    return all_samples

q3c_save_results(q3_c)