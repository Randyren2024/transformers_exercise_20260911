import math
import os
import re
import time
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Step 19
# Training Parallelism vs Autoregressive Inference
#
# Main idea:
#   Training   -> the whole sequence is processed in parallel.
#   Inference  -> generated tokens are produced one by one.
#
# During training we use the real next token as the target.
# This is often called teacher forcing.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare corpus...")
    with urlopen(DATA_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(text)
else:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

print(f"Corpus characters: {len(text)}")

# Word-level tokenizer, matching the previous experiments.
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

all_tokens = tokenize(text)
MAX_TOKENS = 50_000
tokens = all_tokens[:MAX_TOKENS]

vocab = sorted(set(tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}
ids = torch.tensor([stoi[token] for token in tokens], dtype=torch.long)

print(f"Corpus tokens used: {len(tokens)}")
print(f"Vocabulary size: {len(vocab)}")

TRAIN_RATIO = 0.9
split = int(len(ids) * TRAIN_RATIO)
train_ids = ids[:split]
val_ids = ids[split:]

print(f"Train tokens: {len(train_ids)}")
print(f"Validation tokens: {len(val_ids)}")

# ============================================================
# Hyperparameters
# ============================================================
block_size = 64
batch_size = 16
embedding_dim = 64
heads = 4
layers = 2
ffn_dim = 256
max_steps = 1200
learning_rate = 3e-4

print("\nModel:")
print(f"block_size = {block_size}")
print(f"batch_size = {batch_size}")
print(f"embedding_dim = {embedding_dim}")
print(f"heads = {heads}")
print(f"layers = {layers}")
print(f"ffn_dim = {ffn_dim}")


def get_batch(data, batch_size, block_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_heads, num_layers, ffn_dim):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ffn_dim, block_size) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")

        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    @torch.no_grad()
    def next_token_logits(self, idx):
        context = idx[:, -self.block_size :]
        logits, _ = self(context)
        return logits[:, -1, :]


model = TinyGPT(
    vocab_size=len(vocab),
    block_size=block_size,
    d_model=embedding_dim,
    num_heads=heads,
    num_layers=layers,
    ffn_dim=ffn_dim,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {params}")

# ============================================================
# Part 1: One training batch
# ============================================================
print("\nPart 1: Training processes many next-token predictions at once")

x_demo, y_demo = get_batch(train_ids, batch_size, block_size)
with torch.no_grad():
    logits_demo, loss_demo = model(x_demo, y_demo)

print(f"Input shape:  {list(x_demo.shape)}")
print(f"Target shape: {list(y_demo.shape)}")
print(f"Logits shape: {list(logits_demo.shape)}")
print(f"One batch contains {batch_size * block_size} next-token prediction positions.")

# Pick one row so the relationship is easy to inspect.
demo_row = 0
input_tokens = [itos[int(i)] for i in x_demo[demo_row]]
target_tokens = [itos[int(i)] for i in y_demo[demo_row]]

print("\nExample sequence:")
print("Input :", " ".join(input_tokens[:12]), "...")
print("Target:", " ".join(target_tokens[:12]), "...")
print("Each target token is the next token for the corresponding input position.")

# ============================================================
# Part 2: Train
# ============================================================
print("\nTraining:")
model.train()
for step in range(1, max_steps + 1):
    x, y = get_batch(train_ids, batch_size, block_size)

    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    if step == 1 or step % 200 == 0:
        model.eval()
        with torch.no_grad():
            _, train_loss = model(x, y)
            vx, vy = get_batch(val_ids, batch_size, block_size)
            _, val_loss = model(vx, vy)
        model.train()
        print(
            f"Step {step:4d} | train {train_loss.item():.4f} | "
            f"val {val_loss.item():.4f}"
        )

# ============================================================
# Part 3: Inference is autoregressive
# ============================================================
print("\nPart 3: Inference generates one token at a time")
model.eval()


def encode_prompt(prompt):
    prompt_tokens = tokenize(prompt)
    unknown = [t for t in prompt_tokens if t not in stoi]
    if unknown:
        raise ValueError(f"Unknown prompt tokens: {unknown}")
    return torch.tensor([[stoi[t] for t in prompt_tokens]], dtype=torch.long, device=DEVICE)


@torch.no_grad()
def generate(prompt, new_tokens=12):
    idx = encode_prompt(prompt)
    print(f"\nPrompt: {prompt}")
    print("Generation trace:")

    for step in range(new_tokens):
        # The model predicts only the next token from the current context.
        logits = model.next_token_logits(idx)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)

        token = itos[int(next_id.item())]
        print(f"step {step + 1:2d}: chose -> {token}")

    return " ".join(itos[int(i)] for i in idx[0])


try:
    generated = generate("the king", new_tokens=12)
    print("\nFinal generated sequence:")
    print(generated)
except ValueError as e:
    print(f"Generation skipped: {e}")

# ============================================================
# Final summary
# ============================================================
print("\nStep 19 complete.")
print("Training: one forward pass produces predictions for many positions in parallel.")
print("Inference: each newly generated token becomes part of the next input, so generation is autoregressive.")
print("This is why training is highly parallelizable while token-by-token generation is inherently sequential.")
