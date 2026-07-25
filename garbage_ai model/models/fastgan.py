import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "Channels must be divisible by 2 for GLU"
        return x[:, :nc//2] * torch.sigmoid(x[:, nc//2:])

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise

class SLE(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.channel_mult = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_c, out_c, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        return x_high * self.channel_mult(x_low)

class GeneratorBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_c, out_c * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c * 2),
            GLU(),
            NoiseInjection(out_c),
            nn.Conv2d(out_c, out_c * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c * 2),
            GLU(),
            NoiseInjection(out_c)
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.nz = nz
        
        # Initial block: nz -> 8x8 resolution
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 16),
            GLU(),
            nn.ConvTranspose2d(ngf * 8, ngf * 8 * 2, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 8 * 2),
            GLU()
        )
        
        # Generator blocks
        self.feat_16 = GeneratorBlock(ngf * 8, ngf * 4)  # 8x8 -> 16x16
        self.feat_32 = GeneratorBlock(ngf * 4, ngf * 2)  # 16x16 -> 32x32
        self.feat_64 = GeneratorBlock(ngf * 2, ngf)      # 32x32 -> 64x64
        self.feat_128 = GeneratorBlock(ngf, ngf // 2)    # 64x64 -> 128x128
        
        # Skip-Layer Channel Attention (SLE) modules
        self.sle_8_to_128 = SLE(ngf * 8, ngf // 2)
        self.sle_16_to_64 = SLE(ngf * 4, ngf)
        
        # Output layer to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf // 2, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        # Expect z shape: [batch_size, nz, 1, 1]
        x8 = self.init_block(z)
        x16 = self.feat_16(x8)
        x32 = self.feat_32(x16)
        
        x64 = self.feat_64(x32)
        x64 = self.sle_16_to_64(x16, x64)
        
        x128 = self.feat_128(x64)
        x128 = self.sle_8_to_128(x8, x128)
        
        return self.to_rgb(x128)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        
        # Input layer: 128x128 -> 64x64
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
        
        # Feature extractors
        self.feat_32 = DownBlock(ndf, ndf * 2)      # 64x64 -> 32x32
        self.feat_16 = DownBlock(ndf * 2, ndf * 4)  # 32x32 -> 16x16
        self.feat_8 = DownBlock(ndf * 4, ndf * 8)   # 16x16 -> 8x8
        self.feat_4 = DownBlock(ndf * 8, ndf * 16)  # 8x8 -> 4x4
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # Reconstruction decoders for self-supervised learning
        self.reconstruct_8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Discriminator forward pass
        x64 = self.input_layer(x)
        x32 = self.feat_32(x64)
        x16 = self.feat_16(x32)
        x8 = self.feat_8(x16)
        x4 = self.feat_4(x8)
        
        validity = self.classifier(x4).view(-1)
        
        # Return validity and features for self-supervised crop reconstruction
        return validity, x8

    def reconstruct_features(self, x8_feat):
        # Reconstruct the original image from the 8x8 feature map to enforce structure
        return self.reconstruct_8(x8_feat)
