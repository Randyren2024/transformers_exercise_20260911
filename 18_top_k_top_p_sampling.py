import math
import os
import re
import urllib.request
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 1. Reuse the same Tiny Shakespeare corpus as Step 16
# ------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare corpus...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved to: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    corpus_text = f.read()

# Word-level tokenizer, same basic idea as Step 16.
tokens = re.findall(r"\w+|[^\w\s]", corpus_text.lower())
max_corpus_tokens = 50000
tokens = tokens[:max_corpus_tokens]

vocab = sorted(set(tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}
encoded = torch.tensor([stoi[token] for token in tokens], dtype=torch.long)

split_index = int(0.9 * len(encoded))
train_data = encoded[:split_index]
val_data = encoded[split_index:]

print(f"Device: cpu")
print(f"Corpus characters: {len(corpus_text)}")
print(f"Corpus tokens used: {len(encoded)}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Train tokens: {len(train_data)}")
print(f"Validation tokens: {len(val_data)}")


# ------------------------------------------------------------
# 2. Tiny Transformer (same architecture as Step 16)
# ------------------------------------------------------------
block_size = 64
batch_size = 16
embedding_dim = 64
num_heads = 4
num_layers = 2
ffn_dim = 256
max_steps = 1500
learning_rate = 3e-4


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
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
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.shape
        if seq_len > block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size={block_size}")

        positions = torch.arange(seq_len)
        x = self.token_embedding(idx) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss


def get_batch(data):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


model = TinyGPT(
    vocab_size=len(vocab),
    block_size=block_size,
    d_model=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ffn_dim=ffn_dim,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nModel:")
print(f"block_size = {block_size}")
print(f"batch_size = {batch_size}")
print(f"embedding_dim = {embedding_dim}")
print(f"heads = {num_heads}")
print(f"layers = {num_layers}")
print(f"ffn_dim = {ffn_dim}")
print(f"Total trainable parameters: {parameter_count}")

x, y = get_batch(train_data)
logits, loss = model(x, y)
print("\nShape check:")
print(f"Input: {list(x.shape)}")
print(f"Target: {list(y.shape)}")
print(f"Logits: {list(logits.shape)}")
print(f"Initial loss: {loss.item():.4f}")


# ------------------------------------------------------------
# 3. Train
# ------------------------------------------------------------
print("\nTraining:")
for step in range(1, max_steps + 1):
    model.train()
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch(val_data)
            _, val_loss = model(val_x, val_y)

        print(
            f"Step {step:4d} | train {loss.item():.4f} | val {val_loss.item():.4f}"
        )


# ------------------------------------------------------------
# 4. Sampling helpers
# ------------------------------------------------------------
def apply_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def top_k_filter(logits, k):
    """Keep only the k highest-logit tokens; set the rest to -inf."""
    if k is None or k <= 0:
        return logits

    k = min(k, logits.size(-1))
    top_values, _ = torch.topk(logits, k)
    threshold = top_values[..., -1, None]
    filtered = logits.masked_fill(logits < threshold, float("-inf"))
    return filtered


def top_p_filter(logits, p):
    """Keep the smallest set of highest-probability tokens whose cumulative probability reaches p."""
    if p is None or p >= 1.0:
        return logits
    if p <= 0:
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens once cumulative probability exceeds p.
    sorted_remove = cumulative_probs > p
    # Always keep the first token above the threshold.
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    filtered = logits.clone()
    filtered[sorted_indices[sorted_remove]] = float("-inf")
    return filtered


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = apply_temperature(logits, temperature)

    if top_k is not None:
        logits = top_k_filter(logits, top_k)

    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


# ------------------------------------------------------------
# 5. Generate text with different sampling strategies
# ------------------------------------------------------------
def encode_prompt(prompt):
    prompt_tokens = re.findall(r"\w+|[^\w\s]", prompt.lower())
    unknown = [token for token in prompt_tokens if token not in stoi]
    if unknown:
        raise ValueError(f"Unknown prompt tokens: {unknown}")
    return torch.tensor([[stoi[token] for token in prompt_tokens]], dtype=torch.long)


def decode(ids):
    return " ".join(itos[int(i)] for i in ids)


def generate(
    prompt,
    max_new_tokens=30,
    temperature=1.0,
    top_k=None,
    top_p=None,
    seed=42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    idx = encode_prompt(prompt)
    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]
        model.eval()
        with torch.no_grad():
            logits, _ = model(context)
        next_logits = logits[:, -1, :].squeeze(0)
        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        idx = torch.cat([idx, torch.tensor([[next_id]])], dim=1)

    return decode(idx[0])


prompt = "the king"
print("\nGeneration comparison:")
print(f"\nPrompt: {prompt}\n")

experiments = [
    ("Temperature only", 1.0, None, None),
    ("Top-k (k=10)", 1.0, 10, None),
    ("Top-k (k=50)", 1.0, 50, None),
    ("Top-p (p=0.80)", 1.0, None, 0.80),
    ("Top-p (p=0.95)", 1.0, None, 0.95),
    ("Top-k=50 + Top-p=0.95", 1.0, 50, 0.95),
]

for name, temperature, top_k, top_p in experiments:
    try:
        text = generate(
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=42,
        )
        print(f"{name}:")
        print(text)
        print()
    except ValueError as exc:
        print(f"{name}: skipped ({exc})\n")

print("Step 18 complete.")
