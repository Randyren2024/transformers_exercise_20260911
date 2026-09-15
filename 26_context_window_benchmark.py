import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Step 26: Context Window Benchmark
#
# Goal:
#   Keep the Transformer architecture fixed and vary only the
#   context length (block_size).
#
# We measure:
#   1. Attention matrix shape
#   2. Attention-score elements
#   3. Approximate attention-score memory
#   4. Forward-pass timing on CPU
#   5. Scaling relative to sequence length
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

d_model = 64
heads = 4
ffn_dim = 256
batch_size = 16
vocab_size = 1000

CONTEXTS = [16, 32, 64]


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

    def forward(self, x, return_attention=False):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf"),
        )
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)

        if return_attention:
            return out, weights
        return out


class TinyContextModel(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.attn = MultiHeadSelfAttention(d_model, heads, block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, return_attention=False):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]

        if return_attention:
            attn_out, weights = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            logits = self.lm_head(x)
            return logits, weights

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return self.lm_head(x)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_context(block_size, repeats=5):
    model = TinyContextModel(vocab_size, block_size).to(DEVICE)
    model.eval()
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=DEVICE)

    with torch.no_grad():
        logits, weights = model(x, return_attention=True)

    # Warm-up.
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats

    score_elements_per_batch = heads * block_size * block_size
    total_score_elements = batch_size * score_elements_per_batch
    score_memory_mb = total_score_elements * 4 / (1024 ** 2)

    return {
        "block_size": block_size,
        "params": parameter_count(model),
        "input_shape": list(x.shape),
        "logits_shape": list(logits.shape),
        "attention_shape": list(weights.shape),
        "attention_elements_per_batch": total_score_elements,
        "score_memory_mb": score_memory_mb,
        "time_ms": elapsed * 1000,
    }


print("=" * 88)
print("Step 26: Context Window Benchmark")
print("=" * 88)
print("Fixed model: d_model=64, heads=4, ffn_dim=256, batch=16")
print(f"Context lengths tested: {CONTEXTS}")
print()

results = []
for context in CONTEXTS:
    result = benchmark_context(context)
    results.append(result)
    print(f"Context = {context}")
    print(f"  Parameters:                  {result['params']:,}")
    print(f"  Input shape:                 {result['input_shape']}")
    print(f"  Logits shape:                {result['logits_shape']}")
    print(f"  Attention shape:             {result['attention_shape']}")
    print(f"  Attention score elements:   {result['attention_elements_per_batch']:,}")
    print(f"  Score memory (float32):      {result['score_memory_mb']:.3f} MB")
    print(f"  Forward time:                {result['time_ms']:.3f} ms")
    print()

print("=" * 88)
print("Scaling summary")
print("=" * 88)
print("context | attention elements | relative T^2 | score memory | forward time ms")
print("--------+--------------------+--------------+--------------+-----------------")
base = results[0]
for r in results:
    t2_ratio = (r["block_size"] / base["block_size"]) ** 2
    print(
        f"{r['block_size']:7d} | "
        f"{r['attention_elements_per_batch']:18,d} | "
        f"{t2_ratio:12.1f}x | "
        f"{r['score_memory_mb']:12.3f} MB | "
        f"{r['time_ms']:15.3f}"
    )

print()
print("Key observations:")
print("1. The Transformer architecture and parameter count stay fixed while context length changes.")
print("2. The attention score matrix is T x T for each head, so its size grows approximately with T^2.")
print("3. Doubling context length therefore makes the attention-score matrix about 4x larger.")
print("4. The actual forward-time increase is implementation- and hardware-dependent, so it will not necessarily be exactly 4x.")
print("5. This is the core reason long-context attention is computationally and memory intensive.")
print("6. KV Cache helps autoregressive inference by avoiding repeated K/V projections, but it does not remove the need to attend over the growing history.")

print("\nStep 26 complete.")
