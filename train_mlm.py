import torch
import torch.nn as nn
from torch.optim import Adam
from encoder import MLMModel

PAD = "[PAD]"
MASK = "[mask]"
UNK = "[UNK]"

data = [
    ("Transformers use [MASK] attention", "Transformers use self attention"),
    ("Mars is called the [MASK] planet", "Mars is called the red planet"),
    ("Online learning improves [MASK] access", "Online learning improves educational access"),
    ("Exercise improves [MASK] health", "Exercise improves mental health"),
    ("Cricket is a [MASK] sport", "Cricket is a popular sport"),
    ("Python is a [MASK] language", "Python is a programming language"),
    ("Neural networks have [MASK] layers", "Neural networks have hidden layers"),
    ("Trees reduce [MASK] pollution", "Trees reduce air pollution"),
    ("Robots perform [MASK] tasks", "Robots perform repetitive tasks"),
    ("Solar power is a [MASK] source", "Solar power is a renewable source"),
]

def tokenize(s):
    return s.lower().split()

def build_vocab(pairs):
    words = {PAD, MASK, UNK}
    for a, b in pairs:
        for w in tokenize(a) + tokenize(b):
            words.add(w)
    vocab = sorted(list(words))
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return stoi, itos

stoi, itos = build_vocab(data)

def encode(s, max_len=12):
    toks = tokenize(s)
    ids = [stoi.get(t, stoi[UNK]) for t in toks]
    ids = ids[:max_len]
    attn = [1]*len(ids)
    while len(ids) < max_len:
        ids.append(stoi[PAD])
        attn.append(0)
    return torch.tensor(ids), torch.tensor(attn)

def make_batch(pairs, max_len=12):
    X, A, Y, mask_pos = [], [], [], []
    for masked, target in pairs:
        x, a = encode(masked, max_len)
        y, _ = encode(target, max_len)
        
        mp = (x == stoi[MASK]).nonzero(as_tuple=False).squeeze(-1)
        X.append(x); A.append(a); Y.append(y); mask_pos.append(mp)
    return torch.stack(X), torch.stack(A), torch.stack(Y), mask_pos

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLMModel(vocab_size=len(stoi), d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=12).to(device)

X, A, Y, mask_pos = make_batch(data, max_len=12)
X, A, Y = X.to(device), A.to(device), Y.to(device)

opt = Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(300):
    model.train()
    opt.zero_grad()
    logits = model(X, attn_mask=A)  

    loss = 0.0
    count = 0
    for i in range(X.size(0)):
        for p in mask_pos[i]:
            loss = loss + criterion(logits[i, p], Y[i, p])
            count += 1
    loss = loss / max(count, 1)

    loss.backward()
    opt.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1} | loss={loss.item():.4f}")


model.eval()
with torch.no_grad():
    logits = model(X, attn_mask=A)
    preds = logits.argmax(dim=-1)  

def decode(ids):
    words = []
    for i in ids:
        w = itos[int(i)]
        if w == PAD: 
            continue
        words.append(w)
    return " ".join(words)

print("\n--- Predictions ---")
for i, (masked, target) in enumerate(data):
    out_ids = X[i].clone()
    mp = (out_ids == stoi[MASK]).nonzero(as_tuple=False).squeeze(-1)
    for p in mp:
        out_ids[p] = preds[i, p]
    print(f"IN : {masked}")
    print(f"OUT: {decode(out_ids)}")
    print(f"GT : {target}\n")


torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos}, "results/mlm_model.pt")
print("Saved: results/mlm_model.pt")
