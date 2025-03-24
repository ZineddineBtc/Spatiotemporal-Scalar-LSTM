import torch
import torch.nn as nn

from .convlstm import ConvLSTM

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 3D Conv Path
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # ConvLSTM Path
        self.convlstm = ConvLSTM(
            input_channels=in_channels,
            hidden_channels=out_channels
        )
        
        self.attn = nn.Sequential(
            nn.Conv3d(2*out_channels, 1, kernel_size=1),  # Projects to (B,1,T,H,W)
            nn.Sigmoid()
        )

    def forward(self, x):
        # 3D Conv Path
        out_3d = self.conv3d(x)  # (B, C_out, T, H, W)
        
        # ConvLSTM Path
        x_convlstm = x.permute(0, 2, 1, 3, 4)  # (B, T, C_in, H, W)
        out_convlstm = self.convlstm(x_convlstm)  # (B, C_out, T, H, W) - No permute needed!
        
        combined = torch.cat([out_convlstm, out_3d], dim=1)  # (B, 2*C_out, T, H, W)
        alpha = self.attn(combined)  # (B,1,T,H,W)
        return alpha * out_3d + (1 - alpha) * out_convlstm

class STN(nn.Module):
    def __init__(self, T, H, W):
        super().__init__()
        
        self.block1 = FusionBlock(1, 3)
        self.block2 = FusionBlock(3, 3)
        
        self.final_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*T*H*W, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T, H, W)
        x = self.block1(x)
        x = self.block2(x)
        return self.final_fc(x)

