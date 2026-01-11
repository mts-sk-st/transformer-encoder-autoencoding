import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadSelfAttention
from positional_encoding import PositionalEncoding

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        a = self.attn(x, attn_mask=attn_mask)
        x = self.ln1(x + a)
        f = self.ff(x)
        x = self.ln2(x + f)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=64, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, input_ids, attn_mask=None):
        x = self.emb(input_ids)     
        x = self.pos(x)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x

class MLMModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=64):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attn_mask=None):
        h = self.encoder(input_ids, attn_mask=attn_mask)
        logits = self.lm_head(h)  
        return logits

class ClassifierModel(nn.Module):
    def __init__(self, vocab_size, num_classes=2, d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=64):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attn_mask=None):
        h = self.encoder(input_ids, attn_mask=attn_mask)  
        
        if attn_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = attn_mask.unsqueeze(-1)  
            pooled = (h * m).sum(dim=1) / (m.sum(dim=1).clamp(min=1))
        return self.classifier(pooled)
