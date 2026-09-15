import math
import time

import torch
import torch.nn.functional as F


# ============================================================
# Step 30: Flash Attention - core idea
#
# Goal:
#   Compare standard attention that materializes the full T x T
#   attention-score matrix with a tiled, online-softmax version.
#
# This is an educational implementation, not the optimized CUDA
# FlashAttention kernel used in production.
#
# Core idea:
#   Naive attention:
#       scores = Q @ K^T
#       weights = softmax(scores)
#       output = weights @ V
#   The full scores/weights matrix is materialized.
#
#   Tiled/online attention:
#       Process K/V in blocks.
#       Keep only running softmax statistics (m, l) and output.
#       Never materialize the full T x T attention matrix.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

B = 1
H = 4
T = 256
D = 32
BLOCK = 64

print("=" * 88)
print("Step 30: Flash Attention - core idea")
print("=" * 88)
print(
    f"Shape: batch={B}, heads={H}, sequence={T}, head_dim={D}, "
    f"KV block={BLOCK}"
)
print("This is a teaching implementation; it is not the production CUDA kernel.")

# Same Q/K/V for both implementations.
q = torch.randn(B, H, T, D, device=DEVICE)
k = torch.randn(B, H, T, D, device=DEVICE)
v = torch.randn(B, H, T, D, device=DEVICE)


def causal_mask(q_len, k_len, device):
    # For the equal-length self-attention case used here.
    return torch.tril(torch.ones(q_len, k_len, device=device, dtype=torch.bool))


@torch.no_grad()
def naive_attention(q, k, v):
    """Standard causal attention that materializes the full score matrix."""
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    mask = causal_mask(scores.size(-2), scores.size(-1), scores.device)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = weights @ v
    return out, scores, weights


@torch.no_grad()
def tiled_online_attention(q, k, v, block_size):
    """Educational Flash-style attention using tiling + online softmax.

    For each query row we maintain:
      m = running maximum logit
      l = running sum of exp(logit - m)
      o = running weighted value sum in the current scaling

    When a new K/V block arrives, the old running statistics are rescaled
    so the result matches one global softmax, without storing the full T x T
    score matrix.
    """
    B, H, Tq, D = q.shape
    Tk = k.size(-2)
    scale = 1.0 / math.sqrt(D)

    out = torch.zeros(B, H, Tq, D, device=q.device, dtype=q.dtype)
    running_m = torch.full(
        (B, H, Tq), float("-inf"), device=q.device, dtype=q.dtype
    )
    running_l = torch.zeros(B, H, Tq, device=q.device, dtype=q.dtype)
    running_o = torch.zeros(B, H, Tq, D, device=q.device, dtype=q.dtype)

    for start in range(0, Tk, block_size):
        end = min(start + block_size, Tk)
        k_block = k[:, :, start:end, :]
        v_block = v[:, :, start:end, :]

        # Only a T x BLOCK tile is materialized here.
        scores = torch.matmul(q, k_block.transpose(-2, -1)) * scale

        # Causal mask: query i may only attend to key j <= i.
        q_positions = torch.arange(Tq, device=q.device)[:, None]
        k_positions = torch.arange(start, end, device=q.device)[None, :]
        allowed = k_positions <= q_positions
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))

        block_m = scores.max(dim=-1).values
        new_m = torch.maximum(running_m, block_m)

        # exp(old_m - new_m) safely rescales the previously accumulated state.
        old_scale = torch.exp(running_m - new_m)
        block_scale = torch.exp(scores - new_m.unsqueeze(-1))

        new_l = running_l * old_scale + block_scale.sum(dim=-1)
        new_o = (
            running_o * old_scale.unsqueeze(-1)
            + torch.matmul(block_scale, v_block)
        )

        running_m = new_m
        running_l = new_l
        running_o = new_o

    out = running_o / running_l.unsqueeze(-1)
    return out


print("\nPart 1: Naive attention memory")
score_elements = B * H * T * T
score_bytes = score_elements * 4  # float32
print(f"Full attention-score elements: {score_elements:,}")
print(f"Full score memory (float32):   {score_bytes / 1024**2:.3f} MB")

tile_elements = B * H * T * BLOCK
# The tiled implementation also keeps Q, K, V and running state; this number
# isolates the score-tile size to make the central idea visible.
tile_bytes = tile_elements * 4
print(f"One score tile elements:        {tile_elements:,}")
print(f"One score tile memory (float32):{tile_bytes / 1024**2:.3f} MB")
print(f"Tile score-memory ratio:         {tile_elements / score_elements:.2%}")

print("\nPart 2: Correctness check")
naive_out, _, _ = naive_attention(q, k, v)
tiled_out = tiled_online_attention(q, k, v, BLOCK)
max_diff = (naive_out - tiled_out).abs().max().item()
mean_diff = (naive_out - tiled_out).abs().mean().item()
print(f"Max absolute difference:  {max_diff:.6e}")
print(f"Mean absolute difference: {mean_diff:.6e}")
print("Outputs numerically close:", torch.allclose(naive_out, tiled_out, atol=1e-5, rtol=1e-5))

print("\nPart 3: Timing benchmark")

def benchmark(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats

naive_time = benchmark(lambda: naive_attention(q, k, v)[0])
tiled_time = benchmark(lambda: tiled_online_attention(q, k, v, BLOCK))
print(f"Naive attention: {naive_time * 1000:.3f} ms")
print(f"Tiled online:    {tiled_time * 1000:.3f} ms")
print(f"Relative speed:  {naive_time / tiled_time:.2f}x")
print("Note: this educational Python/PyTorch version is not expected to be faster on CPU.")
print("The production FlashAttention win comes from fused GPU kernels, tiling, and IO-aware execution.")

print("\nPart 4: Why this is called memory-efficient")
print("Naive path:")
print("  Q @ K^T -> materialize full [B, H, T, T] scores")
print("  softmax -> materialize another large intermediate")
print("  weights @ V")
print("Tiled path:")
print("  process small K/V blocks")
print("  update online softmax statistics (running max/sum/output)")
print("  discard the block and move to the next one")
print("  never store the complete T x T attention matrix")

print("\nPart 5: Scaling intuition")
print("The mathematical attention still compares each query with the history.")
print("Flash-style tiling mainly reduces the need to materialize and repeatedly move large intermediates to/from memory.")
print("This is why the core idea is better described as an IO/memory-efficiency optimization, not a change to attention semantics.")

print("\nStep 30 complete.")
