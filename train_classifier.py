import torch
import torch.nn as nn
from torch.optim import Adam
from encoder import ClassifierModel

PAD = "[PAD]"
UNK = "[UNK]"


samples = [
    ("i love transformers", 1),
    ("this model is great", 1),
    ("attention works well", 1),
    ("i hate bugs", 0),
    ("this is terrible", 0),
    ("training is hard", 0),
]

def tokenize(s): return s.lower().split()

def build_vocab(samples):
    words = {PAD, UNK}
    for s, _ in samples:
        for w in tokenize(s):
            words.add(w)
    vocab = sorted(list(words))
    stoi = {w:i for i,w in enumerate(vocab)}
    return stoi

stoi = build_vocab(samples)

def encode(s, max_len=6):
    toks = tokenize(s)
    ids = [stoi.get(t, stoi[UNK]) for t in toks][:max_len]
    attn = [1]*len(ids)
    while len(ids) < max_len:
        ids.append(stoi[PAD]); attn.append(0)
    return torch.tensor(ids), torch.tensor(attn)

X, A, Y = [], [], []
for s, label in samples:
    x, a = encode(s)
    X.append(x); A.append(a); Y.append(label)

X = torch.stack(X)
A = torch.stack(A)
Y = torch.tensor(Y)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClassifierModel(vocab_size=len(stoi), num_classes=2, d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=6).to(device)

X, A, Y = X.to(device), A.to(device), Y.to(device)

opt = Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    model.train()
    opt.zero_grad()
    logits = model(X, attn_mask=A)
    loss = criterion(logits, Y)
    loss.backward()
    opt.step()
    if (epoch+1) % 50 == 0:
        acc = (logits.argmax(dim=-1) == Y).float().mean().item()
        print(f"Epoch {epoch+1} | loss={loss.item():.4f} | acc={acc:.2f}")
