import math

import torch


# ============================================================
# Step 31: Flash Attention - memory scaling intuition
#
# Goal:
#   Make the difference between naive attention and tiled/Flash-style
#   attention visible as sequence length T grows.
#
# Important:
#   This step focuses on memory scaling, not on implementing a faster
#   production FlashAttention kernel.
#
# Future project direction:
#   The exercise is gradually moving toward a real small language model.
#   The eventual goal is NOT to make the largest possible model.
#   Instead, we want a moderate-sized modern text corpus that can be
#   trained on the user's CPU, with enough data and a suitable tokenizer
#   to produce a genuinely usable basic conversational model.
#
#   We will therefore keep two principles:
#     1. Learn the important Transformer engineering ideas first.
#     2. Later optimize the dataset/model/training loop around available CPU.
#
#   A practical later path is:
#       modern text corpus
#          -> cleaning / deduplication
#          -> train/validation split
#          -> real tokenizer (BPE/SentencePiece-style)
#          -> small decoder-only Transformer
#          -> causal language-model pretraining
#          -> instruction/conversation fine-tuning
#          -> basic chat inference
#
#   The model size will be chosen from actual CPU benchmark results rather
#   than guessed in advance.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

B = 1
H = 4
D = 32
BLOCK = 64
DTYPE_BYTES = 4  # float32

SEQUENCE_LENGTHS = [256, 512, 1024, 2048]

print("=" * 96)
print("Step 31: Flash Attention - memory scaling")
print("=" * 96)
print(f"Config: batch={B}, heads={H}, head_dim={D}, tile={BLOCK}, dtype=float32")
print("Focus: how T changes the materialized attention-score memory")


def mb(byte_count):
    return byte_count / 1024**2


print("\nPart 1: Theoretical memory scaling")
print(
    f"{'T':>6} {'Naive T×T':>14} {'Naive MB':>12} "
    f"{'Tile T×B':>14} {'Tile MB':>12} {'Ratio':>10}"
)
print("-" * 72)

for T in SEQUENCE_LENGTHS:
    naive_elements = B * H * T * T
    naive_bytes = naive_elements * DTYPE_BYTES

    tile_elements = B * H * T * BLOCK
    tile_bytes = tile_elements * DTYPE_BYTES

    ratio = tile_elements / naive_elements

    print(
        f"{T:6d} "
        f"{naive_elements:14,} "
        f"{mb(naive_bytes):12.3f} "
        f"{tile_elements:14,} "
        f"{mb(tile_bytes):12.3f} "
        f"{ratio:9.2%}"
    )

print("\nInterpretation:")
print("  Naive attention stores a score matrix proportional to T × T -> O(T²).")
print("  Tiled attention stores one score tile proportional to T × BLOCK -> O(T).")
print("  BLOCK stays fixed at 64 in this teaching example.")

print("\nPart 2: Relative scaling")
base_T = SEQUENCE_LENGTHS[0]
base_naive = base_T * base_T
base_tile = base_T * BLOCK

print(f"{'T':>6} {'Naive relative':>16} {'Tile relative':>16}")
print("-" * 42)
for T in SEQUENCE_LENGTHS:
    naive_relative = (T * T) / base_naive
    tile_relative = (T * BLOCK) / base_tile
    print(f"{T:6d} {naive_relative:16.1f}x {tile_relative:16.1f}x")

print("\nThis is the key pattern:")
print("  T: 256 -> 512 -> 1024 -> 2048")
print("  Naive score memory: 1x -> 4x -> 16x -> 64x")
print("  One tiled block:    1x -> 2x -> 4x -> 8x")

print("\nPart 3: Materialize actual score tensors")
print("We will allocate only the score matrix itself, not a full Transformer, to keep the experiment focused.")

for T in SEQUENCE_LENGTHS:
    score_shape = (B, H, T, T)
    score_elements = math.prod(score_shape)
    expected_mb = mb(score_elements * DTYPE_BYTES)

    try:
        scores = torch.empty(score_shape, dtype=torch.float32)
        actual_bytes = scores.numel() * scores.element_size()
        print(
            f"T={T:4d}: shape={list(score_shape)}, "
            f"allocated={mb(actual_bytes):.3f} MB, "
            f"expected={expected_mb:.3f} MB"
        )
        del scores
    except RuntimeError as exc:
        print(f"T={T:4d}: allocation failed -> {exc}")
        break

print("\nPart 4: Why the tile memory still grows")
print("A tile is not a magic constant-size buffer here:")
print("  score tile shape = [batch, heads, T, BLOCK]")
print("So it grows linearly with T.")
print("The important difference is that it avoids storing the complete [B, H, T, T] matrix.")

print("\nPart 5: What this means for a future CPU language model")
print("For our future small GPT-like model, longer context is valuable but expensive.")
print("We therefore need to choose context length together with dataset size, model size, and CPU budget.")
print("The target is a useful basic conversational model, not a huge model for its own sake.")

print("\nStep 31 complete.")
