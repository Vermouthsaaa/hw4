import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 根据二维网格生成位置编码
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

# 根据一维网格生成位置编码
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# 生成位置编码
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.linear = nn.Linear(hidden_size, 6 * hidden_size)

    def forward(self, x, c):
        c = F.silu(c)
        c = self.linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)

        h = self.norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        attn_output, _ = self.attn(h.transpose(0, 1), h.transpose(0, 1), h.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)
        x = x + gate_msa.unsqueeze(1) * attn_output

        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.linear2 = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, c):
        c = F.silu(c)
        c = self.linear1(c)
        shift, scale = c.chunk(2, dim=1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.linear2(x)
        return x

class DiT(nn.Module):
    def __init__(self, input_shape, patch_size, hidden_size, num_heads, num_layers, num_classes, cfg_dropout_prob):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.cfg_dropout_prob = cfg_dropout_prob

        self.pos_embed = nn.Parameter(torch.tensor(get_2d_sincos_pos_embed(hidden_size, input_shape[1] // patch_size), dtype=torch.float32), requires_grad=False)

        self.patch_embed = nn.Conv2d(input_shape[0], hidden_size, kernel_size=patch_size, stride=patch_size)
        
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.y_embed = nn.Embedding(num_classes + 1, hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, patch_size, input_shape[0])

    def patchify_flatten(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def unpatchify(self, x):
        B, L, D = x.shape
        H = W = int(np.sqrt(L))
        P = self.patch_size
        x = x.transpose(1, 2).reshape(B, 4, H * P, W * P)
        return x

    def dropout_classes(self, y, cfg_dropout_prob):
        mask = torch.rand(y.shape[0]) < cfg_dropout_prob
        y[mask] = self.num_classes
        return y

    def forward(self, x, y, t):
        x = self.patchify_flatten(x)
        x = x + self.pos_embed

        t = self.t_embed(t)
         
        if self.training:
            y = self.dropout_classes(y, self.cfg_dropout_prob)
        y = self.y_embed(y)
        c = t + y

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x