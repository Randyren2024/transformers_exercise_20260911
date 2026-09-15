import math
import os
import re
import time
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Step 20
# KV Cache Benchmark
#
# Goal:
#   Compare ordinary autoregressive generation with generation
#   that caches the Key/Value tensors from previous tokens.
#
# This file intentionally does NOT train the model. The purpose
# is to isolate the inference computation and make the cache
# mechanism easy to see.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ------------------------------------------------------------
# Load the same Tiny Shakespeare vocabulary used previously.
# ------------------------------------------------------------
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


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


tokens = tokenize(text)[:50_000]
vocab = sorted(set(tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

print(f"Vocabulary size: {len(vocab)}")


# ============================================================
# Model
# ============================================================
BLOCK_SIZE = 64
EMBEDDING_DIM = 64
HEADS = 4
LAYERS = 2
FFN_DIM = 256


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward_full(self, x):
        """Ordinary causal self-attention for a whole sequence."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        return self.out(self.combine_heads(out)), k, v

    def forward_cached(self, x_new, past_k, past_v):
        """Process only the new token and append its K/V to the cache."""
        q, k_new, v_new = self.qkv(x_new).chunk(3, dim=-1)

        q = self.split_heads(q)
        k_new = self.split_heads(k_new)
        v_new = self.split_heads(v_new)

        if past_k is None:
            k_all = k_new
            v_all = v_new
        else:
            k_all = torch.cat([past_k, k_new], dim=2)
            v_all = torch.cat([past_v, v_new], dim=2)

        scores = q @ k_all.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        out = weights @ v_all

        return self.out(self.combine_heads(out)), k_all, v_all


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

    def forward_full(self, x):
        attn_out, k, v = self.attn.forward_full(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, k, v

    def forward_cached(self, x_new, past_k, past_v):
        attn_out, k_all, v_all = self.attn.forward_cached(
            self.ln1(x_new), past_k, past_v
        )
        x_new = x_new + attn_out
        x_new = x_new + self.ffn(self.ln2(x_new))
        return x_new, k_all, v_all


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.blocks = nn.ModuleList(
            [TransformerBlock(EMBEDDING_DIM, HEADS, FFN_DIM) for _ in range(LAYERS)]
        )
        self.ln_f = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, vocab_size)

    @torch.no_grad()
    def forward_full(self, idx):
        B, T = idx.shape
        if T > BLOCK_SIZE:
            raise ValueError(f"Sequence length {T} exceeds block size {BLOCK_SIZE}")

        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]

        caches = []
        for block in self.blocks:
            x, k, v = block.forward_full(x)
            caches.append((k, v))

        logits = self.lm_head(self.ln_f(x))
        return logits, caches

    @torch.no_grad()
    def forward_cached(self, idx_new, position, caches):
        """Run exactly one new token through every Transformer block."""
        if position >= BLOCK_SIZE:
            raise ValueError(f"Position {position} exceeds block size {BLOCK_SIZE}")

        x = self.token_embedding(idx_new) + self.position_embedding(
            torch.tensor([position], device=idx_new.device)
        )[None, :, :]

        new_caches = []
        for block, (past_k, past_v) in zip(self.blocks, caches):
            x, k_all, v_all = block.forward_cached(x, past_k, past_v)
            new_caches.append((k_all, v_all))

        logits = self.lm_head(self.ln_f(x))
        return logits, new_caches


model = TinyGPT(len(vocab)).to(DEVICE)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params}")
print(
    f"Architecture: d_model={EMBEDDING_DIM}, heads={HEADS}, "
    f"layers={LAYERS}, block_size={BLOCK_SIZE}"
)


# ------------------------------------------------------------
# Prompt helpers
# ------------------------------------------------------------
def encode_prompt(prompt):
    prompt_tokens = tokenize(prompt)
    unknown = [t for t in prompt_tokens if t not in stoi]
    if unknown:
        raise ValueError(f"Unknown prompt tokens: {unknown}")
    ids = [stoi[t] for t in prompt_tokens]
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def decode(ids):
    return " ".join(itos[int(i)] for i in ids)


# ============================================================
# Naive generation
# ============================================================
@torch.no_grad()
def generate_naive(prompt_ids, new_tokens):
    idx = prompt_ids.clone()
    qkv_token_count = 0

    for _ in range(new_tokens):
        context = idx[:, -BLOCK_SIZE:]
        logits, _ = model.forward_full(context)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)

        # Every token in the current context gets a new Q/K/V projection
        # in every layer.
        qkv_token_count += context.size(1)

    return idx, qkv_token_count


# ============================================================
# KV-cache generation
# ============================================================
@torch.no_grad()
def generate_cached(prompt_ids, new_tokens):
    idx = prompt_ids.clone()
    prompt_len = idx.size(1)

    # Process the prompt once and keep each block's K/V tensors.
    logits, caches = model.forward_full(idx)
    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    # Initial prompt required one Q/K/V projection per prompt token.
    qkv_token_count = prompt_len

    for step in range(new_tokens):
        idx = torch.cat([idx, next_id], dim=1)
        position = prompt_len + step

        # Only this newly added token is projected into Q/K/V.
        logits, caches = model.forward_cached(next_id, position, caches)
        qkv_token_count += 1
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    return idx, qkv_token_count


# ============================================================
# Benchmark helpers
# ============================================================
def benchmark(fn, prompt_ids, new_tokens, repeats=3):
    # Warm up once so the first call does not dominate the timing.
    fn(prompt_ids, new_tokens)

    times = []
    result = None
    qkv_tokens = None

    for _ in range(repeats):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result, qkv_tokens = fn(prompt_ids, new_tokens)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return result, qkv_tokens, sum(times) / len(times)


print("\n============================================================")
print("KV Cache benchmark")
print("============================================================")
print("The model is intentionally untrained: this isolates inference cost.")

prompt = "the king"
prompt_ids = encode_prompt(prompt)
new_tokens = 24

print(f"Prompt: {prompt}")
print(f"Prompt tokens: {prompt_ids.size(1)}")
print(f"New tokens: {new_tokens}")
print(f"Repeats per benchmark: 3")

# Use a fixed prompt and the same model weights for both methods.
naive_result, naive_qkv_tokens, naive_time = benchmark(
    generate_naive, prompt_ids, new_tokens
)
cached_result, cached_qkv_tokens, cached_time = benchmark(
    generate_cached, prompt_ids, new_tokens
)

print("\nGenerated sequence (naive):")
print(decode(naive_result[0]))

print("\nGenerated sequence (KV cache):")
print(decode(cached_result[0]))

same_output = torch.equal(naive_result, cached_result)
print(f"\nOutputs identical: {same_output}")

print("\nQ/K/V projection token work (all layers):")
print(f"Naive total context tokens processed: {naive_qkv_tokens}")
print(f"KV-cache token projections:              {cached_qkv_tokens}")
print(f"Projection work reduction:               {1 - cached_qkv_tokens / naive_qkv_tokens:.1%}")

print("\nTiming:")
print(f"Naive generation:    {naive_time:.6f} s")
print(f"KV-cache generation: {cached_time:.6f} s")
print(f"Speedup:              {naive_time / cached_time:.2f}x")

print("\n============================================================")
print("Scaling experiment")
print("============================================================")
print("Same prompt; longer generation means more repeated history for the naive method.")

print("\nnew_tokens | naive_QKV_tokens | cache_QKV_tokens | theoretical QKV reduction")
print("-----------+------------------+------------------+------------------------")

for n in [8, 16, 24, 32]:
    _, nq, _ = benchmark(generate_naive, prompt_ids, n, repeats=1)
    _, cq, _ = benchmark(generate_cached, prompt_ids, n, repeats=1)
    reduction = 1 - cq / nq
    print(f"{n:10d} | {nq:16d} | {cq:16d} | {reduction:22.1%}")

print("\nStep 20 complete.")
print("KV cache stores previous K/V tensors so old tokens do not need new Q/K/V projections.")
print("The cache does not remove attention against the history; it removes repeated projection work.")
