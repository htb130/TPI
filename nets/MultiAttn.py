import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size, dim_feedforward=2048, dropout=0.1):
        super(MultiAttnLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size

        # Multi-head attention with the given number of heads
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layer for feedforward
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # Split input into windows along the time dimension
        batch_size, seq_len, embed_dim = src.size()
        assert embed_dim == self.d_model, "Embedding dimension mismatch"

        # Process each window separately
        windowed_outputs = []
        for i in range(0, seq_len, self.window_size):
            window_src = src[:, i:i + self.window_size, :]

            # Attention requires sequence length in the first dimension
            window_src = window_src.permute(1, 0, 2)
            attn_output, _ = self.self_attn(window_src, window_src, window_src)
            attn_output = attn_output.permute(1, 0, 2)
            windowed_outputs.append(attn_output)

        # Concatenate all windows back along the sequence dimension
        attn_output = torch.cat(windowed_outputs, dim=1)

        # Add & Norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward with Add & Norm
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src
