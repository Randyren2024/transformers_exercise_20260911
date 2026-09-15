import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Step 32: CPU training efficiency benchmark
#
# Goal:
#   Measure how fast this computer can train small GPT-like models.
#   The result will later help us decide whether to train locally,
#   use the user's Colab T4 GPU, or use a hybrid workflow.
#
# Important:
#   This is a benchmark, not a language-model training run.
#   We intentionally use synthetic/random token IDs so that data
#   loading does not become the bottleneck.
#
# We measure:
#   1. parameter count
#   2. step time
#   3. training tokens / second
#   4. approximate tokens / minute
#
# We test several small model sizes so we can see the tradeoff between
# model capacity and CPU speed.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

# Let PyTorch use the available CPU threads.
# We print the number so the benchmark is reproducible enough for comparison.
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass

DEVICE = "cpu"
VOCAB_SIZE = 5000
BATCH_SIZE = 8
SEQ_LEN = 64
WARMUP_STEPS = 5
BENCHMARK_STEPS = 20

print("=" * 96)
print("Step 32: CPU training efficiency benchmark")
print("=" * 96)
print(f"Device: {DEVICE}")
print(f"CPU threads used by PyTorch: {torch.get_num_threads()}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Tokens processed per training step: {BATCH_SIZE * SEQ_LEN:,}")
print("This benchmark uses random data so disk/data loading is not part of the measurement.")


class TinyGPT(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ffn_dim, max_seq_len=SEQ_LEN):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "q_proj": nn.Linear(d_model, d_model),
                "k_proj": nn.Linear(d_model, d_model),
                "v_proj": nn.Linear(d_model, d_model),
                "o_proj": nn.Linear(d_model, d_model),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, d_model),
                ),
            })
            for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=input_ids.device))
        causal = causal.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            h = block["ln1"](x)
            q = block["q_proj"](h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = block["k_proj"](h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = block["v_proj"](h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores = scores.masked_fill(~causal, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            attn = weights @ v
            attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)

            x = x + block["o_proj"](attn)
            x = x + block["ffn"](block["ln2"](x))

        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_model(name, d_model, n_heads, n_layers, ffn_dim):
    torch.manual_seed(SEED)

    model = TinyGPT(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    params = count_parameters(model)

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    print("\n" + "-" * 96)
    print(f"Model: {name}")
    print(
        f"d_model={d_model}, heads={n_heads}, layers={n_layers}, "
        f"ffn={ffn_dim}, parameters={params:,}"
    )

    model.train()

    # Warmup allows PyTorch to settle before timing.
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        loss.backward()
        optimizer.step()

    start = time.perf_counter()

    for _ in range(BENCHMARK_STEPS):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start
    avg_step = elapsed / BENCHMARK_STEPS
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    tokens_per_second = tokens_per_step / avg_step
    tokens_per_minute = tokens_per_second * 60

    print(f"Final benchmark loss: {loss.item():.4f}")
    print(f"Average step time:     {avg_step * 1000:.2f} ms")
    print(f"Tokens / second:       {tokens_per_second:,.0f}")
    print(f"Tokens / minute:       {tokens_per_minute:,.0f}")

    return {
        "name": name,
        "parameters": params,
        "step_ms": avg_step * 1000,
        "tokens_per_second": tokens_per_second,
        "tokens_per_minute": tokens_per_minute,
    }


# These are deliberately modest models. They are not the final model.
# The purpose is to locate the useful CPU training range.
configs = [
    ("Small", 64, 4, 2, 256),
    ("Medium", 96, 4, 3, 384),
    ("Larger", 128, 4, 4, 512),
]

results = []
for config in configs:
    results.append(benchmark_model(*config))

print("\n" + "=" * 96)
print("Summary")
print("=" * 96)
print(f"{'Model':<10} {'Params':>12} {'Step ms':>12} {'tok/s':>12} {'tok/min':>14}")
print("-" * 96)
for r in results:
    print(
        f"{r['name']:<10} "
        f"{r['parameters']:>12,} "
        f"{r['step_ms']:>12.2f} "
        f"{r['tokens_per_second']:>12,.0f} "
        f"{r['tokens_per_minute']:>14,.0f}"
    )

fastest = max(results, key=lambda x: x["tokens_per_second"])
slowest = min(results, key=lambda x: x["tokens_per_second"])

print("\nInterpretation:")
print("  Larger models perform more computation per token, so CPU throughput falls.")
print("  The useful number for our future project is tokens/second, not just parameter count.")
print("  Once we know the throughput, we can estimate how long 1M / 10M / 50M training tokens would take.")
print(
    f"  Fastest benchmark: {fastest['name']} at "
    f"{fastest['tokens_per_second']:,.0f} tok/s"
)
print(
    f"  Slowest benchmark: {slowest['name']} at "
    f"{slowest['tokens_per_second']:,.0f} tok/s"
)

print("\nFuture training strategy:")
print("  Local CPU -> experiments, debugging, tokenizer work, small training runs")
print("  Colab GPU -> longer pretraining runs when CPU training becomes impractical")
print("  Hybrid -> write code locally in VS Code/WSL, run the same code on Colab T4 when needed")
print("  The Colab T4 can therefore be our fallback rather than forcing the CPU to do everything.")
print("  Colab GPU availability and quotas can vary over time, so the actual runtime should be checked when used.")

print("\nStep 32 complete.")
