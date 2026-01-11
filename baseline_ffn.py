import torch
import torch.nn as nn

class BaselineFFN(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attn_mask=None):
        x = self.emb(input_ids)  
        if attn_mask is None:
            pooled = x.mean(dim=1)
        else:
            m = attn_mask.unsqueeze(-1)
            pooled = (x*m).sum(dim=1) / (m.sum(dim=1).clamp(min=1))
        return self.net(pooled)
