import torch
import torch.nn as nn

# ConvLSTM returning the full hidden sequence
class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell that processes one timestep at a time.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Combine the 4 gating operations into one convolution
        self.conv = nn.Conv2d(
            in_channels = input_channels + hidden_channels,
            out_channels = 4 * hidden_channels,
            kernel_size = self.kernel_size,
            padding = self.padding,
            bias = True
        )
        
    def forward(self, x, h_prev, c_prev):
        """
        x:      (batch, input_channels, height, width)
        h_prev: (batch, hidden_channels, height, width)
        c_prev: (batch, hidden_channels, height, width)
        """
        combined = torch.cat([x, h_prev], dim=1)  # (B, in+hidden, H, W)
        gates = self.conv(combined)               # (B, 4*hidden, H, W)
        
        # Split into 4 different gates
        in_gate, forget_gate, g_gate, out_gate = torch.split(
            gates, self.hidden_channels, dim=1
        )
        
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        g_gate = torch.tanh(g_gate)
        out_gate = torch.sigmoid(out_gate)
        
        c_cur = forget_gate * c_prev + in_gate * g_gate
        h_cur = out_gate * torch.tanh(c_cur)
        
        return h_cur, c_cur

class ConvLSTM(nn.Module):
    """
    A multi-step ConvLSTM that processes a sequence of T frames.
    Returns the entire hidden sequence => (B, hidden_channels, T, H, W).
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.cell = ConvLSTMCell(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        
    def forward(self, x):
        """
        x: (B, T, C, H, W)
        We'll produce the entire hidden sequence: (B, hidden, T, H, W).
        """
        B, T, C, H, W = x.shape
        
        h_cur = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        c_cur = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        
        outputs = []
        for t in range(T):
            x_t = x[:, t]        # (B, C, H, W)
            h_cur, c_cur = self.cell(x_t, h_cur, c_cur)
            outputs.append(h_cur.unsqueeze(2))  # => (B, hidden, 1, H, W)
        
        # Concatenate along time dimension => (B, hidden, T, H, W)
        out = torch.cat(outputs, dim=2)
        return out
