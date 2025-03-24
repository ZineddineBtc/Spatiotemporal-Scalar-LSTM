import torch.nn as nn

from .convlstm import ConvLSTM

# FusionBlock: 3D Convs + ConvLSTM + Elementwise (Average) Fusion
class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        
        # 3D Conv pipeline, each with ReLU
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # ConvLSTM pipeline
        self.convlstm = ConvLSTM(
            input_channels = in_channels,
            hidden_channels = out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 3D conv path => (B, out_channels, T, H, W)
        out_3d = self.conv3d(x)
        
        # ConvLSTM path => reorder x to (B, T, C, H, W)
        x_lstm = x.permute(0, 2, 1, 3, 4)
        out_convlstm = self.convlstm(x_lstm)  # => (B, out_channels, T, H, W)
        
        # Fuse by averaging
        fused = 0.5 * (out_3d + out_convlstm)
        return fused  # (B, out_channels, T, H, W)


class STN(nn.Module):
    def __init__(self, T,  H, W):
        super(STN, self).__init__()
        
        self.block1 = FusionBlock(in_channels=1, out_channels=3)
        self.block2 = FusionBlock(in_channels=3, out_channels=3)
        self.final_fc = nn.Sequential(
            nn.Linear(3*T*H*W, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # => (B, 1, T, H, W)
        
        out_block = self.block1(x)   # => (B, 3, T, H, W)
        out_block = self.block2(out_block)
        
        # Flatten everything: (B, 6, T, H, W) => (B, 6*T*H*W)
        B, C, T, H, W = out_block.shape
        flat = out_block.view(B, -1)  # => (B, 6*T*H*W)
        
        # MLP => (B, 1)
        out = self.final_fc(flat)
        return out.squeeze(1)


