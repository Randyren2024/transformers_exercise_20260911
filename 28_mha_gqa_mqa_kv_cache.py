import math

# ============================================================
# Step 28: MHA vs GQA vs MQA
#
# Goal:
#   Keep the model width and number of query heads fixed, and only
#   change the number of KV heads. This isolates the effect on KV cache.
#
# MHA: Query heads = 4, KV heads = 4
# GQA: Query heads = 4, KV heads = 2
# MQA: Query heads = 4, KV heads = 1
# ============================================================

D_MODEL = 64
Q_HEADS = 4
HEAD_DIM = D_MODEL // Q_HEADS
LAYERS = 2
BATCH = 1
CONTEXT = 1024

assert Q_HEADS % 2 == 0

CONFIGS = [
    ("MHA", 4),
    ("GQA", 2),
    ("MQA", 1),
]


def cache_elements(batch, layers, context, kv_heads, head_dim):
    # K and V are both cached.
    return batch * layers * context * kv_heads * head_dim * 2


def mb(num_elements, bytes_per_value):
    return num_elements * bytes_per_value / (1024 ** 2)


def print_attention_shape(name, kv_heads):
    # In grouped-query attention, each KV head serves a group of Q heads.
    q_per_kv = Q_HEADS // kv_heads
    print(f"{name}:")
    print(f"  Query heads: {Q_HEADS}")
    print(f"  KV heads:    {kv_heads}")
    print(f"  Queries per KV head: {q_per_kv}")
    print(
        f"  K/V cache shape per layer: "
        f"[batch={BATCH}, context={CONTEXT}, kv_heads={kv_heads}, head_dim={HEAD_DIM}]"
    )


print("=" * 90)
print("Step 28: MHA vs GQA vs MQA - KV Cache experiment")
print("=" * 90)
print(
    f"Fixed: d_model={D_MODEL}, query_heads={Q_HEADS}, "
    f"head_dim={HEAD_DIM}, layers={LAYERS}, batch={BATCH}, context={CONTEXT}"
)
print("Only KV heads change. This isolates KV-cache memory impact.")

print("\nPart 1: Attention structures")
for name, kv_heads in CONFIGS:
    print_attention_shape(name, kv_heads)

print("\nPart 2: KV Cache memory at 1,024 tokens")
print("type | KV heads | elements | FP32 MB | FP16 MB | relative FP16")
print("-----+----------+----------+----------+----------+--------------")

mha_fp16 = None
for name, kv_heads in CONFIGS:
    elements = cache_elements(BATCH, LAYERS, CONTEXT, kv_heads, HEAD_DIM)
    fp32 = mb(elements, 4)
    fp16 = mb(elements, 2)
    if mha_fp16 is None:
        mha_fp16 = fp16
    relative = fp16 / mha_fp16
    print(
        f"{name:4s} | {kv_heads:8d} | {elements:8,d} | "
        f"{fp32:8.3f} | {fp16:8.3f} | {relative:12.2f}x"
    )

print("\nPart 3: Memory savings vs MHA")
for name, kv_heads in CONFIGS:
    elements = cache_elements(BATCH, LAYERS, CONTEXT, kv_heads, HEAD_DIM)
    fp16 = mb(elements, 2)
    savings = (1 - fp16 / mha_fp16) * 100
    print(f"{name:4s}: FP16 KV cache = {fp16:.3f} MB | saving vs MHA = {savings:.1f}%")

print("\nPart 4: Scaling with context length")
contexts = [128, 256, 512, 1024]
for name, kv_heads in CONFIGS:
    print(f"\n{name} (KV heads={kv_heads})")
    base = None
    for context in contexts:
        elements = cache_elements(BATCH, LAYERS, context, kv_heads, HEAD_DIM)
        fp16 = mb(elements, 2)
        if base is None:
            base = fp16
        print(f"  context={context:4d} -> {fp16:.3f} MB ({fp16 / base:.1f}x)")

print("\nPart 5: Same cache, different batch sizes")
for name, kv_heads in CONFIGS:
    elements = cache_elements(16, LAYERS, CONTEXT, kv_heads, HEAD_DIM)
    fp16 = mb(elements, 2)
    print(f"batch=16 | {name:3s} | FP16 KV cache = {fp16:.3f} MB")

print("\nPart 6: Conceptual comparison")
print("MHA: every query head has its own K/V head.")
print("GQA: several query heads share each K/V head.")
print("MQA: all query heads share one K/V head.")
print("Fewer KV heads -> proportionally smaller KV cache.")
print("The experiment measures cache size, not model quality or latency.")

print("\nStep 28 complete.")
