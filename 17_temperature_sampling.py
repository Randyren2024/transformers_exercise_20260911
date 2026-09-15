import math
import re
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 17. Temperature sampling
#
# Same tiny GPT architecture as step 16.
# The important change is ONLY the generation method:
# greedy argmax vs. temperature sampling.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"
DATA_PATH = f"{DATA_DIR}/tinyshakespeare.txt"


# ------------------------------------------------------------
# Load corpus
# ------------------------------------------------------------
import os
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare corpus...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved to: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Keep the same word-level tokenizer idea as step 16.
tokens = re.findall(r"\w+|[^\w\s]", text.lower())
max_tokens = 50000
tokens = tokens[:max_tokens]

vocab = sorted(set(tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}
encoded = torch.tensor([stoi[token] for token in tokens], dtype=torch.long)

train_size = int(0.9 * len(encoded))
train_data = encoded[:train_size]
val_data = encoded[train_size:]

print(f"Corpus characters: {len(text)}")
print(f"Corpus tokens used: {len(tokens)}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Train tokens: {len(train_data)}")
print(f"Validation tokens: {len(val_data)}")


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------
block_size = 64
batch_size = 16
embedding_dim = 64
num_heads = 4
num_layers = 2
ffn_dim = 256

print("\nModel:")
print(f"block_size = {block_size}")
print(f"batch_size = {batch_size}")
print(f"embedding_dim = {embedding_dim}")
print(f"heads = {num_heads}")
print(f"layers = {num_layers}")
print(f"ffn_dim = {ffn_dim}")


# ------------------------------------------------------------
# Batch helper
# ------------------------------------------------------------
def get_batch(data):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


# ------------------------------------------------------------
# Multi-head self-attention
# ------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(out)


# ------------------------------------------------------------
# Transformer block
# ------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, block_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, block_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ------------------------------------------------------------
# Tiny GPT
# ------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    ffn_dim,
                    block_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.shape
        if seq_len > block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {block_size}")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss


model = TinyGPT(len(vocab)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# ------------------------------------------------------------
# Quick shape/loss check
# ------------------------------------------------------------
x, y = get_batch(train_data)
logits, loss = model(x, y)
print("\nShape check:")
print(f"Input: {list(x.shape)}")
print(f"Target: {list(y.shape)}")
print(f"Logits: {list(logits.shape)}")
print(f"Initial loss: {loss.item():.4f}")


# ------------------------------------------------------------
# Train briefly
# ------------------------------------------------------------
print("\nTraining:")
for step in range(1, 1501):
    x, y = get_batch(train_data)

    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0:
        model.eval()
        with torch.no_grad():
            _, train_loss = model(*get_batch(train_data))
            _, val_loss = model(*get_batch(val_data))
        model.train()
        print(
            f"Step {step:4d} | train {train_loss.item():.4f} | "
            f"val {val_loss.item():.4f}"
        )


# ------------------------------------------------------------
# Tokenize a prompt
# ------------------------------------------------------------
def encode_prompt(prompt):
    prompt_tokens = re.findall(r"\w+|[^\w\s]", prompt.lower())
    unknown = [token for token in prompt_tokens if token not in stoi]
    if unknown:
        raise ValueError(f"Unknown prompt tokens: {unknown}")
    return torch.tensor([[stoi[token] for token in prompt_tokens]], dtype=torch.long, device=DEVICE)


# ------------------------------------------------------------
# Greedy generation
# ------------------------------------------------------------
def generate_greedy(prompt, max_new_tokens=40):
    idx = encode_prompt(prompt)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)

    return decode(idx[0].tolist())


# ------------------------------------------------------------
# Temperature sampling
#
# temperature < 1.0 -> sharper distribution
# temperature = 1.0 -> original distribution
# temperature > 1.0 -> flatter distribution
# ------------------------------------------------------------
def generate_temperature(prompt, temperature=1.0, max_new_tokens=40):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    idx = encode_prompt(prompt)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)

        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return decode(idx[0].tolist())


# ------------------------------------------------------------
# Decode tokens back to text
# ------------------------------------------------------------
def decode(ids):
    pieces = [itos[i] for i in ids]
    text_out = ""

    for piece in pieces:
        if re.match(r"\w+$", piece):
            if text_out and not text_out.endswith((" ", "\n")):
                text_out += " "
            text_out += piece
        else:
            text_out += piece

    return text_out


# ------------------------------------------------------------
# Compare generation methods
# ------------------------------------------------------------
model.eval()

prompt = "the king"
print("\nGeneration comparison:")
print(f"\nPrompt: {prompt}")

with torch.no_grad():
    print("\nGreedy (argmax):")
    print(generate_greedy(prompt))

    for temperature in [0.5, 1.0, 1.5]:
        # Reset the random seed so the comparison is reproducible.
        torch.manual_seed(SEED)
        print(f"\nTemperature = {temperature}:")
        print(generate_temperature(prompt, temperature=temperature))

print("\nStep 17 complete.")
