"""
microgpt, rewritten with PyTorch. Same data, same model, same hyperparameters.
"""
import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
random.seed(42); torch.manual_seed(42)

# Dataset
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt', 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

# Tokenizer
uchars = sorted(set(''.join(docs)))
BOS, vocab_size = len(uchars), len(uchars) + 1

# Model
n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
norm = lambda x: F.rms_norm(x, (n_embd,), eps=1e-5)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # wq, wk, wv in one matrix
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd, bias=False))
    def forward(self, x):                                       # x: (T, n_embd), all positions at once
        T = x.shape[0]
        q, k, v = self.qkv(norm(x)).split(n_embd, dim=-1)
        q, k, v = (t.view(T, n_head, -1).transpose(0, 1) for t in (q, k, v))  # split into heads
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # causal mask replaces the KV cache
        x = x + self.wo(y.transpose(0, 1).reshape(T, n_embd))
        return x + self.mlp(norm(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, idx):
        x = norm(self.wte(idx) + self.wpe(torch.arange(len(idx))))
        return self.lm_head(self.blocks(x))

model = GPT()
for p in model.parameters(): nn.init.normal_(p, std=0.08)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

# Training
num_steps = 1000
opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / num_steps)
for step in range(num_steps):
    tokens = torch.tensor([BOS] + [uchars.index(c) for c in docs[step % len(docs)]] + [BOS])[:block_size + 1]
    loss = F.cross_entropy(model(tokens[:-1]), tokens[1:])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# Inference
print("\n--- inference (new, hallucinated names) ---")
with torch.no_grad():
    for i in range(20):
        idx = [BOS]
        while len(idx) <= block_size:
            probs = F.softmax(model(torch.tensor(idx))[-1] / 0.5, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == BOS: break
            idx.append(nxt)
        print(f"sample {i+1:2d}: {''.join(uchars[t] for t in idx[1:])}")
