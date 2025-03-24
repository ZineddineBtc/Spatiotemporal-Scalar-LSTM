import torch.nn as nn

from .convlstm import ConvLSTM

# FusionBlock: 3D Convs + ConvLSTM + Elementwise (Average) Fusion
class FusionBlock(nn.Module):
    """
    One "fusion" block:
      - 3 sequential 3D Convs (in_channels->...->out_channels)
      - 1 ConvLSTM (in_channels->out_channels)
      - Fusion by averaging the 3Dconv output + ConvLSTM output
      => returns (B, out_channels, T, H, W).
    """
    def __init__(self, in_channels, out_channels, H, W):
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

        # MultiheadAttention
        # We interpret 'embed_dim' as (out_channels * H * W) so that each (T) step is a token.
        self.embed_dim = out_channels * H * W
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=1,
            dropout=0.1,
            batch_first=True  # (B, T, E) input shape
        )

    def forward(self, x):
        """
        x: shape (B, in_channels, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # 3D conv path => (B, out_channels, T, H, W)
        out_3d = self.conv3d(x)
        
        # ConvLSTM path => reorder x to (B, T, C, H, W)
        x_lstm = x.permute(0, 2, 1, 3, 4)
        out_convlstm = self.convlstm(x_lstm)  # => (B, out_channels, T, H, W)
        
        B, C, T, H, W = out_3d.shape 
        out_3d_seq = out_3d.permute(0, 2, 1, 3, 4).reshape(B, T, C * H * W)  
        out_lstm_seq = out_convlstm.permute(0, 2, 1, 3, 4).reshape(B, T, C * H * W)
        
        attn_out, attn_weights = self.cross_attention(
            query=out_3d_seq,  # shape (B, T, E)
            key=out_lstm_seq,  # shape (B, T, E)
            value=out_lstm_seq
        )
        attn_out = attn_out + out_3d_seq
        attn_out = attn_out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        fused = 0.5 * (attn_out + out_3d) 
        return fused

class STN(nn.Module):
    """
    two-block model:
      - First block: in_channels=1 -> out_channels=3
      - Second block: in_channels=3 -> out_channels=3
      - Then flatten the entire (B, 3, T, H, W) to pass into Dense(3*T*H*W)->...->Dense(1)
    """
    def __init__(self, T,  H, W):
        super(STN, self).__init__()
        
        self.block1 = FusionBlock(in_channels=1, out_channels=3, H=H, W=W)
        self.block2 = FusionBlock(in_channels=3, out_channels=3, H=H, W=W)
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
        
        # Flatten everything: (B, 3, T, H, W) => (B, 3*T*H*W)
        B, C, T, H, W = out_block.shape
        flat = out_block.reshape(B, -1)  # Use reshape instead of view
        
        # MLP => (B, 1)
        out = self.final_fc(flat)
        return out.squeeze(1)