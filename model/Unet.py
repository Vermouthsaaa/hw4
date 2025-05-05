import torch
import torch.nn as nn

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period, dtype=torch.float32)) *
                      torch.arange(0, half, dtype=torch.float32) / half)
    device = timesteps.device
    freqs = freqs.to(device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act1 = nn.SiLU()
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.SiLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = self.act1(self.norm1(self.conv1(x)))
        temb = self.temb_proj(temb)
        temb = temb.unsqueeze(-1).unsqueeze(-1)
        h += temb
        h = self.act2(self.norm2(self.conv2(h)))
        x = self.shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, hidden_dims, blocks_per_dim):
        super().__init__()
        temb_channels = hidden_dims[0] * 4
        
        self.temb = nn.Sequential(
            nn.Linear(hidden_dims[0], temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels)
        )
        
        self.input_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        prev_ch = hidden_dims[0]
        down_block_chans = [prev_ch]
        
        for i, hidden_dim in enumerate(hidden_dims):
            for _ in range(blocks_per_dim):
                self.down_blocks.append(ResidualBlock(prev_ch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
                down_block_chans.append(prev_ch)
            if i != len(hidden_dims) - 1:
                self.down_blocks.append(Downsample(prev_ch))
                down_block_chans.append(prev_ch)
                
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(prev_ch, prev_ch, temb_channels),
            ResidualBlock(prev_ch, prev_ch, temb_channels)
        ])
        
        for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
            for j in range(blocks_per_dim + 1):
                self.up_blocks.append(ResidualBlock(prev_ch + down_block_chans.pop(), hidden_dim, temb_channels))
                prev_ch = hidden_dim
                if i and j == blocks_per_dim:
                    self.up_blocks.append(Upsample(prev_ch))
        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=prev_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(prev_ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.temb[0].in_features))
        
        if len(temb.shape) > 2:
            temb = temb.view(temb.size(0), -1)
            
        h = self.input_conv(x)
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, temb)
            else:
                h = block(h)
            hs.append(h)
            
        for block in self.mid_blocks:
            h = block(h, temb)
            
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = block(torch.cat([h, hs.pop()], dim=1), temb)
            else:
                h = block(h)
                
        h = self.out_act(self.out_norm(h))
        out = self.out_conv(h)
        return out