import math

import torch


# ============================================================
# Step 27: KV Cache Memory Scaling
#
# Goal:
#   Measure how KV cache memory grows with context length.
#
# Important distinction:
#   Attention score memory grows roughly with T^2.
#   KV cache memory grows roughly linearly with T.
#
# This script is a theoretical + allocation benchmark. It does not
# train the TinyGPT model and does not require a GPU.
# ============================================================

print("=" * 88)
print("Step 27: KV Cache Memory Scaling")
print("=" * 88)

# Keep the dimensions consistent with our TinyGPT experiments.
batch_size = 1
layers = 2
heads = 4
head_dim = 16  # d_model=64 / 4 heads
context_lengths = [128, 256, 512, 1024]

dtypes = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

print(
    f"Configuration: batch={batch_size}, layers={layers}, "
    f"KV heads={heads}, head_dim={head_dim}"
)
print(f"Context lengths: {context_lengths}")
print("Note: this benchmark measures cache memory independently of block_size.")


def cache_elements(context, kv_heads=heads, dim=head_dim, num_layers=layers, batch=batch_size):
    """Number of scalar elements in K + V for all layers."""
    # K: [B, H_kv, T, D]
    # V: [B, H_kv, T, D]
    return 2 * batch * num_layers * kv_heads * context * dim


def bytes_to_mb(nbytes):
    return nbytes / (1024 ** 2)


print("\nPart 1: Theoretical KV cache memory")
print("context | elements | FP32 MB | FP16 MB | BF16 MB | relative")
print("--------+----------+----------+----------+----------+---------")

base_elements = cache_elements(context_lengths[0])
for context in context_lengths:
    elements = cache_elements(context)
    fp32_mb = bytes_to_mb(elements * 4)
    fp16_mb = bytes_to_mb(elements * 2)
    bf16_mb = bytes_to_mb(elements * 2)
    relative = elements / base_elements
    print(
        f"{context:7d} | {elements:8,d} | {fp32_mb:8.3f} | "
        f"{fp16_mb:8.3f} | {bf16_mb:8.3f} | {relative:7.1f}x"
    )


print("\nPart 2: Memory per generated token")
per_token_elements = cache_elements(1)
print(f"K + V elements per token across all layers: {per_token_elements:,}")
for name, dtype in dtypes.items():
    bytes_per_value = torch.tensor([], dtype=dtype).element_size()
    per_token_bytes = per_token_elements * bytes_per_value
    print(
        f"{name:8s}: {per_token_bytes:,} bytes/token "
        f"({bytes_to_mb(per_token_bytes):.6f} MB/token)"
    )


print("\nPart 3: Actually allocate the cache on CPU")
for name, dtype in dtypes.items():
    print(f"\n{name}:")
    before = cache_elements(context_lengths[0])
    for context in context_lengths:
        # Allocate one K and one V tensor per layer.
        caches = []
        for _ in range(layers):
            k = torch.empty(
                batch_size, heads, context, head_dim, dtype=dtype
            )
            v = torch.empty(
                batch_size, heads, context, head_dim, dtype=dtype
            )
            caches.append((k, v))

        # Touch the tensors so the benchmark represents real allocated storage.
        for k, v in caches:
            k.fill_(0)
            v.fill_(0)

        total_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in caches)
        relative = total_bytes / (before * torch.tensor([], dtype=dtype).element_size())
        print(
            f"  context={context:4d} -> {bytes_to_mb(total_bytes):.3f} MB "
            f"({relative:.1f}x vs {context_lengths[0]})"
        )
        del caches


print("\nPart 4: Compare KV cache growth with attention-score growth")
print("context | KV cache relative | attention-score relative")
print("--------+-------------------+------------------------")
for context in context_lengths:
    kv_relative = context / context_lengths[0]
    attn_relative = (context / context_lengths[0]) ** 2
    print(f"{context:7d} | {kv_relative:17.1f}x | {attn_relative:22.1f}x")


print("\nPart 5: Same 1,024-token context at different batch sizes")
context = 1024
for batch in [1, 4, 16]:
    elements = cache_elements(context, batch=batch)
    fp16_mb = bytes_to_mb(elements * 2)
    print(f"batch={batch:2d} -> FP16 KV cache: {fp16_mb:.3f} MB")


print("\nKey observations:")
print("1. KV cache memory grows approximately linearly with context length T.")
print("2. Attention score memory grows approximately quadratically with T.")
print("3. FP16/BF16 use about half the KV-cache bytes of FP32 for the same shape.")
print("4. More layers, KV heads, head dimension, or batch size all increase KV-cache memory.")
print("5. This is why long-context inference can become memory-bound even when the model weights fit in memory.")
print("6. The current TinyGPT block_size is only 64; larger lengths here are a standalone cache-scaling experiment.")

print("\nStep 27 complete.")
