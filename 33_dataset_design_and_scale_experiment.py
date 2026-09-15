import math


# ============================================================
# Step 33: Modern text dataset design and scale experiment
#
# Goal:
#   Design a realistic training-data strategy for a small GPT-like
#   model that should eventually handle basic conversation.
#
# Important idea:
#   Dataset quality, diversity, and token budget matter alongside
#   model size. We should not simply download the biggest corpus.
#
# This script does not download a large dataset.
# It builds a small synthetic mixture to teach the concepts of:
#   1. data categories
#   2. mixture weights
#   3. train/validation split
#   4. token-budget experiments
#   5. rough training-time estimation using Step 32 throughput
#
# Later we can replace the synthetic examples with selected public
# datasets and keep the same accounting logic.
# ============================================================

print("=" * 96)
print("Step 33: Modern text dataset design and scale experiment")
print("=" * 96)

# ------------------------------------------------------------
# Part 1: Candidate data categories
# ------------------------------------------------------------
# These are intentionally broad categories rather than a claim that
# every final dataset must contain exactly these proportions.
#
# A small conversational base model benefits from more than one style
# of text: general prose/world knowledge, instructional explanations,
# dialogue-like text, and a small amount of code/structured text.
#
# The weights below are a teaching mixture that we can change later.
# ------------------------------------------------------------

categories = {
    "general_web_and_reference": 0.45,
    "educational_and_explanatory": 0.25,
    "stories_and_dialogue": 0.20,
    "code_and_structured_text": 0.10,
}

print("\nPart 1: Example training-data mixture")
print("-" * 96)
for name, weight in categories.items():
    print(f"{name:34s} {weight:6.1%}")
print(f"{'Total':34s} {sum(categories.values()):6.1%}")

# ------------------------------------------------------------
# Part 2: Why not use one huge web corpus?
# ------------------------------------------------------------
print("\nPart 2: Dataset design principles")
print("-" * 96)
print("1. Quality: remove obvious boilerplate, spam, navigation text, and duplicates.")
print("2. Diversity: mix domains and writing styles so the model does not overfit one style.")
print("3. Modernity: prefer relatively recent material when practical, especially for examples")
print("   and conversational language. The base model is still not guaranteed to know current facts.")
print("4. Language: start with English for a controlled experiment; add Chinese later if desired.")
print("5. Token budget: choose a finite target that matches our compute rather than downloading everything.")
print("6. Validation: keep a held-out set that is not used for gradient updates.")

# ------------------------------------------------------------
# Part 3: Token-budget experiments
# ------------------------------------------------------------
# Step 32 measured three CPU configurations. We use the Small model's
# measured throughput as a simple upper-bound planning reference.
# These estimates ignore validation/checkpoint/data-loading overhead.
# ------------------------------------------------------------

THROUGHPUT_TOKENS_PER_SECOND = 20_401  # Step 32 Small-model benchmark

budgets = [
    10_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
]

print("\nPart 3: Training-token budget experiments")
print("-" * 96)
print(f"Reference CPU throughput: {THROUGHPUT_TOKENS_PER_SECOND:,} tok/s")
print()
print(f"{'Tokens':>14s} {'CPU hours':>14s} {'CPU days':>12s} {'Comment':>34s}")
print("-" * 80)

comments = {
    10_000_000: "fast architecture test",
    50_000_000: "small serious experiment",
    100_000_000: "useful small pretraining run",
    250_000_000: "larger CPU commitment",
    500_000_000: "likely better on GPU/Colab",
}

for tokens in budgets:
    seconds = tokens / THROUGHPUT_TOKENS_PER_SECOND
    hours = seconds / 3600
    days = hours / 24
    print(f"{tokens:14,d} {hours:14.2f} {days:12.2f} {comments[tokens]:>34s}")

# ------------------------------------------------------------
# Part 4: Mixture accounting
# ------------------------------------------------------------
TARGET_TOKENS = 100_000_000

print("\nPart 4: Example 100M-token mixture")
print("-" * 96)
print(f"Target training corpus: {TARGET_TOKENS:,} tokens")
print()
print(f"{'Category':34s} {'Weight':>10s} {'Tokens':>18s}")
print("-" * 68)
for name, weight in categories.items():
    count = int(TARGET_TOKENS * weight)
    print(f"{name:34s} {weight:10.1%} {count:18,d}")

# ------------------------------------------------------------
# Part 5: Train/validation split
# ------------------------------------------------------------
# Keep validation separate from training. 5% is enough for a teaching
# and engineering experiment at this scale while leaving most tokens
# for learning.
# ------------------------------------------------------------

TRAIN_FRACTION = 0.95
VAL_FRACTION = 0.05

train_tokens = int(TARGET_TOKENS * TRAIN_FRACTION)
val_tokens = int(TARGET_TOKENS * VAL_FRACTION)

print("\nPart 5: Train / validation split")
print("-" * 96)
print(f"Train:      {train_tokens:,} tokens ({TRAIN_FRACTION:.0%})")
print(f"Validation: {val_tokens:,} tokens ({VAL_FRACTION:.0%})")
print("Validation data must not be used for gradient updates.")

# ------------------------------------------------------------
# Part 6: Dataset-scale thinking
# ------------------------------------------------------------
# A useful way to reason about a small model is to compare its parameter
# count with the number of unique/effective training tokens. The exact
# compute-optimal relationship depends on architecture, optimizer,
# tokenization and training target; here we intentionally use simple
# ratios only as planning tools, not as a universal scaling law.
# ------------------------------------------------------------

parameter_counts = [5_000_000, 10_000_000, 20_000_000, 50_000_000]

token_budgets_for_ratio = [50_000_000, 100_000_000, 250_000_000, 500_000_000]

print("\nPart 6: Parameter-to-token planning ratios")
print("-" * 96)
print("These are planning ratios only; they are not a rule that guarantees good quality.")
print()
print(f"{'Params':>12s} {'Tokens':>14s} {'Tokens / Param':>16s}")
print("-" * 48)
for params in parameter_counts:
    for tokens in token_budgets_for_ratio:
        ratio = tokens / params
        if ratio in (5, 10, 25, 50):
            print(f"{params:12,d} {tokens:14,d} {ratio:16.1f}x")

# ------------------------------------------------------------
# Part 7: Recommended experiment ladder
# ------------------------------------------------------------
print("\nPart 7: Recommended experiment ladder")
print("-" * 96)
print("Phase A: 10M tokens -> confirm tokenizer, training loop, checkpointing and generation.")
print("Phase B: 50M tokens -> compare model sizes and data mixtures.")
print("Phase C: 100M tokens -> first serious small-model pretraining run.")
print("Phase D: 250M+ tokens -> move the longer run to Colab GPU when CPU time becomes inconvenient.")
print("Phase E: after base pretraining -> add a small instruction/dialogue dataset for chat behavior.")

# ------------------------------------------------------------
# Part 8: Candidate public dataset families
# ------------------------------------------------------------
print("\nPart 8: Public dataset families worth evaluating later")
print("-" * 96)
print("FineWeb / FineWeb-Edu: very large cleaned web corpora; use a small selected subset, not the full corpus.")
print("SmolLM-Corpus: curated mixture including Cosmopedia, FineWeb-Edu-Dedup, and Python-Edu.")
print("A small dialogue/instruction set should be added later for conversational behavior rather than expecting")
print("raw pretraining text alone to create a polished assistant.")

# ------------------------------------------------------------
# Part 9: Final direction for our project
# ------------------------------------------------------------
print("\nPart 9: Project direction")
print("-" * 96)
print("Our goal is NOT the largest possible corpus.")
print("Our goal is a manageable, diverse, relatively modern corpus that can train a small model well.")
print("The initial target for experimentation is 50M-100M training tokens.")
print("The final pretraining budget may grow to 250M-500M tokens if Colab GPU makes the run practical.")
print("After pretraining, a separate instruction/dialogue stage will teach the model to respond conversationally.")

print("\nStep 33 complete.")
