import torch.nn as nn

from .slstm import sLSTM

# FusionBlock: 3D Convs + sLSTM + Elementwise (Average) Fusion
class FusionBlock(nn.Module):
    """
    One "fusion" block:
      - 3 sequential 3D Convs (in_channels->...->out_channels)
      - 1 sLSTM (in_channels->out_channels)
      - Fusion by averaging the 3Dconv output + sLSTM output
      => returns (B, out_channels, T, H, W).
    """
    def __init__(self, in_channels, out_channels, H, W, slstm_num_heads):
        super(FusionBlock, self).__init__()
        
        self.out_channels = out_channels

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
        
        # sLSTM Path
        self.lstm_input_size = in_channels * H * W
        self.lstm_output_size = out_channels * H * W
        
        # Pre-projection to expand from in_channels*H*W => out_channels*H*W
        self.pre_linear = nn.Linear(self.lstm_input_size, self.lstm_output_size)
        
        # The new sLSTM uses residual blocks, so its input_size == output_size
        self.slstm = sLSTM(
            input_size=self.lstm_output_size,
            head_size=self.lstm_output_size,  # can choose smaller if needed
            num_heads=slstm_num_heads,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        """
        x: shape (B, in_channels, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # 3D conv path => (B, out_channels, T, H, W)
        out_3d = self.conv3d(x)
        
        # sLSTM Path (NEW)
        B, C_in, T, H, W = x.shape
        # Prepare (B, T, in_channels*H*W)
        x_slstm = x.permute(0, 2, 1, 3, 4).reshape(B, T, -1)  # => (B, T, C_in*H*W)
        # Linear projection up to (B, T, out_channels*H*W)
        x_slstm = self.pre_linear(x_slstm)
        # Pass through the new sLSTM
        slstm_out, _ = self.slstm(x_slstm)  # => (B, T, out_channels*H*W) because batch_first=True
        slstm_out = slstm_out.reshape(B, T, self.out_channels, H, W)
        slstm_out = slstm_out.permute(0, 2, 1, 3, 4)  # => (B, C_out, T, H, W)

        # Fuse by averaging
        fused = 0.5 * (out_3d + slstm_out)
        return fused  # (B, out_channels, T, H, W)


class STN(nn.Module):
    """
    two-block model:
      - First block: in_channels=1 -> out_channels=3
      - Second block: in_channels=3 -> out_channels=6
      - Then flatten the entire (B, 6, T, H, W) to pass into Dense(6)->Dense(4)->Dense(1)->Sigmoid
    """
    def __init__(self, T,  H, W):
        super(STN, self).__init__()
        
        self.block1 = FusionBlock(in_channels=1, out_channels=3, H=H, W=W, slstm_num_heads=4)
        self.block2 = FusionBlock(in_channels=3, out_channels=3, H=H, W=W, slstm_num_heads=2)
        self.final_fc = nn.Sequential(
            nn.Linear(3*T*H*W, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, T, H, W) => e.g. (8, 6, 4, 4).
        We reshape to (B, 1, T, H, W) for block1's in_channels=1.
        """
        x = x.unsqueeze(1)  # => (B, 1, T, H, W)
        
        out_block = self.block1(x)   # => (B, 3, T, H, W)
        out_block = self.block2(out_block)

        # Flatten everything: (B, 6, T, H, W) => (B, 6*T*H*W)
        B, C, T, H, W = out_block.shape
        flat = out_block.view(B, -1)  # => (B, 6*T*H*W)
        
        # MLP => (B, 1)
        out = self.final_fc(flat)
        return out.squeeze(1)


