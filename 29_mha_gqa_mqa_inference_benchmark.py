import math
import time
import torch
import torch.nn.functional as F

# ============================================================
# Step 29: MHA vs GQA vs MQA - inference benchmark
#
# Goal:
#   Keep the model size and workload fixed while changing only
#   the number of KV heads. Measure:
#     - KV cache memory
#     - cached attention latency
#     - tokens/sec
#
# This is a small CPU benchmark for learning, not a production
# performance benchmark. Absolute timings depend on hardware.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed workload.
batch_size = 1
query_heads = 4
head_dim = 32
layers = 4
prompt_len = 128
new_tokens = 64
warmup_steps = 10
benchmark_repeats = 5

CONFIGS = {
    "MHA": 4,
    "GQA": 2,
    "MQA": 1,
}


def sync_device():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def cache_memory_mb(kv_heads, dtype=torch.float16):
    elements = batch_size * layers * prompt_len * kv_heads * head_dim * 2
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    return elements * bytes_per_element / (1024 ** 2)


def init_cache(kv_heads, dtype=torch.float32):
    # Cache stores K and V for all layers.
    shape = (layers, batch_size, kv_heads, prompt_len, head_dim)
    keys = torch.randn(shape, dtype=dtype, device=DEVICE)
    values = torch.randn(shape, dtype=dtype, device=DEVICE)
    return keys, values


def cached_attention_step(q, k_cache, v_cache, kv_heads):
    """One incremental attention step.

    q:         [B, Q_heads, 1, head_dim]
    k_cache:   [B, KV_heads, T, head_dim]
    v_cache:   [B, KV_heads, T, head_dim]

    When KV heads are fewer than Query heads, each KV head is shared
    by an equal number of query heads (GQA/MQA).
    """
    repeat_factor = query_heads // kv_heads
    k = k_cache.repeat_interleave(repeat_factor, dim=1)
    v = v_cache.repeat_interleave(repeat_factor, dim=1)

    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    weights = F.softmax(scores, dim=-1)
    out = weights @ v
    return out


@torch.no_grad()
def benchmark_one(kv_heads):
    # Use a fresh random query/cache for this configuration.
    # The shapes and workload are identical except KV-head count.
    k_cache, v_cache = init_cache(kv_heads)

    total_new_token_time = 0.0

    for step in range(warmup_steps + benchmark_repeats):
        q = torch.randn(
            batch_size, query_heads, 1, head_dim,
            dtype=torch.float32,
            device=DEVICE,
        )

        sync_device()
        start = time.perf_counter()
        for _ in range(new_tokens):
            # Simulate one generated token attending to the current history.
            # We rebuild q each step to represent a new token.
            q = torch.randn(
                batch_size, query_heads, 1, head_dim,
                dtype=torch.float32,
                device=DEVICE,
            )
            for layer in range(layers):
                _ = cached_attention_step(
                    q,
                    k_cache[layer],
                    v_cache[layer],
                    kv_heads,
                )
        sync_device()
        elapsed = time.perf_counter() - start

        if step >= warmup_steps:
            total_new_token_time += elapsed

    avg_total = total_new_token_time / benchmark_repeats
    tokens_per_sec = (new_tokens * layers) / avg_total
    ms_per_token_per_layer = avg_total * 1000 / (new_tokens * layers)

    return avg_total, tokens_per_sec, ms_per_token_per_layer


print("=" * 90)
print("Step 29: MHA vs GQA vs MQA - inference benchmark")
print("=" * 90)
print(f"Device: {DEVICE}")
print(
    f"Fixed workload: batch={batch_size}, query_heads={query_heads}, "
    f"head_dim={head_dim}, layers={layers}, prompt={prompt_len}, new_tokens={new_tokens}"
)
print(f"Warmup={warmup_steps}, repeats={benchmark_repeats}")
print("Only KV heads change.")

print("\nKV cache memory (FP16, prompt only):")
print("type | KV heads | cache MB | relative")
print("-----+----------+----------+---------")
for name, kv_heads in CONFIGS.items():
    mem = cache_memory_mb(kv_heads, torch.float16)
    print(f"{name:4s} | {kv_heads:8d} | {mem:8.3f} | {mem / cache_memory_mb(4, torch.float16):7.2f}x")

print("\nInference benchmark:")
print("type | KV heads | avg time | tokens/sec | ms/token/layer | relative speed")
print("-----+----------+-----------+------------+----------------+---------------")

results = {}
for name, kv_heads in CONFIGS.items():
    avg_total, tok_s, ms_tok_layer = benchmark_one(kv_heads)
    results[name] = (avg_total, tok_s, ms_tok_layer)

mha_tok_s = results["MHA"][1]
for name, kv_heads in CONFIGS.items():
    avg_total, tok_s, ms_tok_layer = results[name]
    print(
        f"{name:4s} | {kv_heads:8d} | {avg_total:9.4f}s | "
        f"{tok_s:10.2f} | {ms_tok_layer:14.4f} | {tok_s / mha_tok_s:13.2f}x"
    )

print("\nInterpretation:")
print("- GQA/MQA reduce KV-cache memory by reducing the number of stored K/V heads.")
print("- This benchmark measures attention-side inference work, not full LLM end-to-end latency.")
print("- Repeat-interleave is intentionally explicit so the head-sharing mechanism is visible.")
print("- CPU timings can be noisy; focus on trends and relative measurements rather than absolute numbers.")
print("- Lower KV-cache memory does not guarantee a proportional speedup because other matrix operations and memory movement remain.")

print("\nStep 29 complete.")
