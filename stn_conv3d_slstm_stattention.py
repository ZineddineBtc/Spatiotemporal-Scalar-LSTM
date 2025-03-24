import torch
import torch.nn as nn
import torch.nn.functional as F


from .slstm import sLSTM

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, nh):
        super().__init__()
        self.out_channels = out_channels

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
        
        # sLSTM Path
        self.lstm_input_size = in_channels * H * W
        self.lstm_output_size = out_channels * H * W
        # Pre-projection to expand from in_channels*H*W => out_channels*H*W
        self.pre_linear = nn.Linear(self.lstm_input_size, self.lstm_output_size)
        # The new sLSTM uses residual blocks, so its input_size == output_size
        self.slstm = sLSTM(
            input_size=self.lstm_output_size,
            head_size=self.lstm_output_size,  # can choose smaller if needed
            num_heads=nh,
            num_layers=1,
            batch_first=True
        )
        
        # Attention mixing
        self.attn = nn.Sequential(
            nn.Conv3d(2 * out_channels, 1, kernel_size=1),  # Projects to (B,1,T,H,W)
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, in_channels, T, H, W)
        Returns: (B, out_channels, T, H, W)
        """
        # 3D Conv Path
        out_3d = self.conv3d(x)  # => (B, C_out, T, H, W)
        
        # sLSTM Path (NEW)
        B, C_in, T, H, W = x.shape
        # Prepare (B, T, in_channels*H*W)
        x_slstm = x.permute(0, 2, 1, 3, 4).reshape(B, T, -1)  # => (B, T, C_in*H*W)

        # Linear projection up to (B, T, out_channels*H*W)
        x_slstm = self.pre_linear(x_slstm)

        # Pass through the new sLSTM
        slstm_out, _ = self.slstm(x_slstm)  # => (B, T, out_channels*H*W) because batch_first=True

        # Reshape back to (B, out_channels, T, H, W)
        slstm_out = slstm_out.reshape(B, T, self.out_channels, H, W)
        slstm_out = slstm_out.permute(0, 2, 1, 3, 4)  # => (B, C_out, T, H, W)

        # Fuse via attention
        combined = torch.cat([slstm_out, out_3d], dim=1)  # => (B, 2*C_out, T, H, W)
        alpha = self.attn(combined)                      # => (B, 1, T, H, W)
        
        return alpha * out_3d + (1 - alpha) * slstm_out

class STN(nn.Module):
    """Final Corrected Model"""
    def __init__(self, T, H, W):
        super().__init__()
        
        self.block1 = FusionBlock(1, 3, H, W, nh=4)
        self.block2 = FusionBlock(3, 3, H, W, nh=2)
        
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
    



