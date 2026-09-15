import tiktoken

# ============================================================
# Step 25: Token efficiency benchmark
#
# Goal:
#   Compare how many tokens different kinds of real-world text use.
#   We measure characters, UTF-8 bytes, tokens, chars/token, and
#   bytes/token using the same production tokenizer.
#
# This is a tokenizer experiment only. No model training happens here.
# ============================================================

ENCODING_NAME = "cl100k_base"
enc = tiktoken.get_encoding(ENCODING_NAME)

SAMPLES = {
    "English": "The window cleaning drone works efficiently at high altitude.",
    "Chinese": "超高层建筑清洗无人机可以提高高空作业安全性。",
    "Chinese + English": "超高层建筑清洗无人机 window cleaning drone",
    "Code": "def clean_window(drone, height): return drone.start(height)",
    "URL": "https://www.partdro.com/products/firefighting_drone/",
    "Numbers": "MTOW 25 kg, payload 10 kg, flight time 30 minutes.",
    "Product specs": "Partdro D15R | window cleaning drone | 20 L water tank",
    "Mixed symbols": "🚁 AI + drones | 100% safe? #cleaning @Partdro",
}


def utf8_bytes(text):
    return len(text.encode("utf-8"))


def analyze(text):
    token_ids = enc.encode(text)
    chars = len(text)
    bytes_count = utf8_bytes(text)
    token_count = len(token_ids)

    return {
        "chars": chars,
        "bytes": bytes_count,
        "tokens": token_count,
        "chars_per_token": chars / token_count if token_count else 0.0,
        "bytes_per_token": bytes_count / token_count if token_count else 0.0,
    }


print("=" * 92)
print("Step 25: Token efficiency benchmark")
print("=" * 92)
print(f"Encoding: {ENCODING_NAME}")
print(f"Vocabulary size: {enc.n_vocab}")

print("\nPart 1: Token efficiency across text types")
print(
    f"{'Type':18s} | {'Chars':>6s} | {'Bytes':>6s} | {'Tokens':>6s} | "
    f"{'Chars/token':>12s} | {'Bytes/token':>12s}"
)
print("-" * 92)

results = {}
for name, text in SAMPLES.items():
    stats = analyze(text)
    results[name] = stats
    print(
        f"{name:18s} | {stats['chars']:6d} | {stats['bytes']:6d} | "
        f"{stats['tokens']:6d} | {stats['chars_per_token']:12.2f} | "
        f"{stats['bytes_per_token']:12.2f}"
    )

print("\nPart 2: Examples with token pieces")
examples = [
    "window cleaning drone",
    "超高层建筑清洗无人机",
    "https://www.partdro.com/products/firefighting_drone/",
    "def clean_window(drone, height): return drone.start(height)",
]

for text in examples:
    token_ids = enc.encode(text)
    pieces = [enc.decode_single_token_bytes(t) for t in token_ids]
    print(f"\nText: {text!r}")
    print(f"Token count: {len(token_ids)}")
    print(f"Pieces: {pieces}")

print("\nPart 3: What 1,000 tokens can roughly carry")
print(
    f"{'Type':18s} | {'Approx chars in 1,000 tokens':>30s} | "
    f"{'Approx UTF-8 bytes in 1,000 tokens':>35s}"
)
print("-" * 92)

for name, stats in results.items():
    approx_chars = stats["chars_per_token"] * 1000
    approx_bytes = stats["bytes_per_token"] * 1000
    print(f"{name:18s} | {approx_chars:30.0f} | {approx_bytes:35.0f}")

print("\nPart 4: Interpretation")
print("1. Token count is not the same as character count or byte count.")
print("2. Different text types have different token efficiency with the same tokenizer.")
print("3. A context window measured in tokens therefore covers different amounts of raw text depending on the content.")
print("4. Token counts are directly relevant to context limits and token-based API pricing.")
print("5. These are empirical measurements for cl100k_base; another tokenizer can produce different results.")

print("\nStep 25 complete.")
