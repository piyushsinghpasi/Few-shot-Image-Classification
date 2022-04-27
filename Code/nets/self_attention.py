import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # self.activation = nn.ReLU()

    def forward(self, src_q, src_mask=None, src_key_padding_mask=None):
        
        # this permute will change as now we have 2D
        # src_q = src_q.permute(1, 0, 2)
        # src_v = src_v.permute(1, 0, 2)
        src_q = src_q.unsqueeze(dim=0).permute(1, 0, 2)
        src_self = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout12(src_self)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        # return src_q
        return src_q.permute(1,0,2).squeeze(dim=0)