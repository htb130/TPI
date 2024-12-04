import torch
from torch import nn
from torch.nn import functional as F

from common.models.transformer.attention import MultiHeadAttention


class FusionLayer(nn.Module):
    def __init__(self, d_model=256, d_k=64, d_v=64, h=4, dropout=.1):
        super(FusionLayer, self).__init__()
        self.global_p = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.self_p = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.self_d = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

        self.g_p = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, v_p, v_d, attention_mask_p, attention_mask_d):
        b_s = v_p.shape[0]
        g_p = self.g_p.expand(b_s, 1, -1)

        g_p_cnn = self.global_p(g_p, v_p, v_p, attention_mask=attention_mask_p)
        p_features = torch.cat([g_p, v_p], dim=1)
        add_mask_p = torch.zeros(b_s, 1, 1, 1).bool().to(p_features.device)
        attention_mask = torch.cat([add_mask_p, attention_mask_p], dim=-1)
        p_att = self.self_p(p_features, p_features, p_features, attention_mask=attention_mask)

        v_p = p_att[:, 1:]

        v_d = self.self_d(v_d, v_d, v_d, attention_mask=attention_mask_d)

        return v_p, v_d

class FusionEncoder(nn.Module):
    def __init__(self, N, device='cuda', d_model=256, d_k=64, d_v=64, h=4, dropout=.1, p_encoder=None):
        super(FusionEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.device = device
        self.p_encoder = p_encoder

        self.layers = nn.ModuleList([FusionLayer(d_model, d_k, d_v, h, dropout) for _ in range(N)])

    def forward(self, v_p, v_d):
        # input (b_s, seq_len)
        attention_mask_p = (torch.sum(torch.abs(v_p), -1) == 0).unsqueeze(1).unsqueeze(1)
        attention_mask_d = (torch.sum(torch.abs(v_d), -1) == 0).unsqueeze(1).unsqueeze(1)
        for l in self.layers:
            v_p = self.p_encoder(v_p)
            v_p, v_d = l(v_p, v_d, attention_mask_p, attention_mask_d)

        return v_p, v_d


class DP_FusionLayer(nn.Module):
    def __init__(self, d_model=256, d_k=64, d_v=64, h=4, dropout=.1):
        super(DP_FusionLayer, self).__init__()
        self.att1 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.att2 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.att3 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)

    def mean_ignore_zeros(self, tensor):
        mask = tensor.abs().sum(dim=2) != 0  # [64, 290]
        mask = mask.unsqueeze(2)  # [64, 290, 1]
        filtered_tensor = tensor * mask  # [64, 290, 256]
        valid_counts = mask.sum(dim=1)  # [64, 1, 256]
        valid_counts = valid_counts + (valid_counts == 0).float()
        mean_features = filtered_tensor.sum(dim=1) / valid_counts  # [64, 256]

        return mean_features

    def forward(self, v_p, v_d, attention_mask_p, attention_mask_d):
        padsize = v_p.size(1) - v_d.size(1)
        t_d = torch.cat([v_d, torch.zeros(v_d.size(0), padsize, v_d.size(-1)).to(v_d.device)], dim=1)
        attention_mask_t_d = (torch.sum(torch.abs(t_d), -1) == 0).unsqueeze(1).unsqueeze(1)
        c_d = self.att1(t_d, v_p, v_p, attention_mask=attention_mask_t_d)
        c_p = self.att2(v_p, t_d, t_d, attention_mask=attention_mask_p)
        f1 = self.mean_ignore_zeros(c_d)
        f2 = self.mean_ignore_zeros(c_p)
        f3 = self.mean_ignore_zeros(v_d)
        f4 = self.mean_ignore_zeros(v_p)

        return f1, f2, f3, f4


class DP_FusionEncoder(nn.Module):
    def __init__(self, N, device='cuda', d_model=256, d_k=64, d_v=64, h=4, dropout=.1):
        super(DP_FusionEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.device = device

        self.layers = nn.ModuleList([DP_FusionLayer(d_model, d_k, d_v, h, dropout) for _ in range(N)])

    def forward(self, v_p, v_d):
        # mask_pad = (input != self.padding_idx).unsqueeze(-1)
        attention_mask_p = (torch.sum(torch.abs(v_p), -1) == 0).unsqueeze(1).unsqueeze(1)
        attention_mask_d = (torch.sum(torch.abs(v_d), -1) == 0).unsqueeze(1).unsqueeze(1)
        for l in self.layers:
            f1, f2, f3, f4 = l(v_p, v_d, attention_mask_p, attention_mask_d)

        return f1, f2, f3, f4


def build_encoder(N, device='cuda', d_model=256, d_k=64, d_v=64, h=4, dropout=.1):
    Encoder = FusionEncoder(N, device, d_model, d_k, d_v, h, dropout)

    return Encoder