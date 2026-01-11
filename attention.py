import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.last_attn = None  

    def forward(self, x, attn_mask=None):
        
        B, T, D = x.shape

        q = self.Wq(x)  
        k = self.Wk(x)
        v = self.Wv(x)

        
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  

        if attn_mask is not None:
            
            mask = attn_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)  
        attn = self.drop(attn)
        self.last_attn = attn.detach()

        out = attn @ v  
        out = out.transpose(1, 2).contiguous().view(B, T, D)  
        out = self.Wo(out)
        return out
