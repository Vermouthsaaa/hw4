from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

# @property
# def latent_shape(self) -> Tuple[int, int, int]:
#     """Size of the encoded representation"""

# def encode(self, x: np.ndarray) -> np.ndarray:
#     """Encode an image x. Note: Channel dim is in dim 1

#     Args:
#         x (np.ndarray, dtype=float32): Image to encode. shape=(batch_size, 3, 32, 32). Values in [-1, 1]

#     Returns:
#         np.ndarray: Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values
#     """

# def decode(self, z: np.ndarray) -> np.ndarray:
#     """Decode an encoded image.

#     Args:
#         z (np.ndarray, dtype=float32): Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values.

#     Returns:
#         np.ndarray: Decoded image. shape=(batch_size, 3, 32, 32). Values in [-1, 1]
#     """


def q3_a(images, vae):
    """
    images: (1000, 32, 32, 3) numpy array in [0, 1], the images to pass through the encoder and decoder of the vae
    vae: a vae model, trained on the relevant dataset

    Returns
    - a numpy array of size (50, 2, 32, 32, 3) of the decoded image in [0, 1] consisting of pairs
      of real and reconstructed images
    - a float that is the scale factor
    """
  
    image = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).cuda()

    # 编码
    latent_space = vae.encode(image)

    # 计算缩放因子
    scale_factor = np.std(latent_space.detach().cpu().numpy().flatten())

    # 解码
    reimages = vae.decode(latent_space)
    
    reimages = reimages.detach().cpu().permute(0, 2, 3, 1).numpy()
    selected_images = np.stack([images[:50], reimages[:50]], axis=1)

    return selected_images, scale_factor

q3a_save_results(q3_a)