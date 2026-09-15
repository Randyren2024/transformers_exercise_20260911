# ============================================================
# Step 24: A real GPT-style tokenizer with tiktoken
#
# Goal:
#   Compare our teaching tokenizers with a production-grade
#   byte-level BPE tokenizer used by OpenAI's tokenizer stack.
#
# Install once if needed:
#   pip install tiktoken
# ============================================================

try:
    import tiktoken
except ImportError:
    raise SystemExit(
        "tiktoken is not installed. Run:\n"
        "    pip install tiktoken\n"
        "then run this file again."
    )


print("=" * 72)
print("Step 24: Real tokenizer experiment with tiktoken")
print("=" * 72)

# cl100k_base is a widely used OpenAI BPE encoding.
# This experiment is about how a production tokenizer behaves,
# not about reproducing any particular chat model exactly.
enc = tiktoken.get_encoding("cl100k_base")

print("Encoding:", enc.name)
print("Vocabulary size:", enc.n_vocab)

examples = [
    "I love cats",
    "The king was speaking to the people.",
    "Romeo and Juliet are famous.",
    "window cleaning drone",
    "超高层建筑清洗无人机",
    "Partdro D15R",
    "https://www.partdro.com/products/firefighting_drone/",
    "print('hello world')",
    "1234567890",
    "hello_world",
    "🚁 AI + drones",
]


print("\n" + "=" * 72)
print("Part 1: Tokenization examples")
print("=" * 72)

for text in examples:
    ids = enc.encode(text)
    pieces = [enc.decode_single_token_bytes(i) for i in ids]
    safe_pieces = [repr(piece) for piece in pieces]

    print("\nText:", repr(text))
    print("Token count:", len(ids))
    print("Token IDs:", ids)
    print("Byte pieces:", safe_pieces)
    print("Decoded:", repr(enc.decode(ids)))


print("\n" + "=" * 72)
print("Part 2: Why byte-level tokenization is useful")
print("=" * 72)

special_cases = {
    "unseen-like word": "antidisestablishmentarianism",
    "mixed case": "Romeo romeo ROMEO",
    "Chinese + English": "你好, world",
    "URL": "https://example.com/a/b?q=123",
    "code": "def hello(name): return f'Hi {name}'",
}

for name, text in special_cases.items():
    ids = enc.encode(text)
    print(f"\n{name}: {text!r}")
    print("Tokens:", len(ids))
    print("Pieces:", [repr(enc.decode_single_token_bytes(i)) for i in ids])


print("\n" + "=" * 72)
print("Part 3: Word-level vs real tokenizer")
print("=" * 72)

comparison_text = "The window cleaning drone works efficiently."
word_level = comparison_text.lower().split()
tiktoken_ids = enc.encode(comparison_text)

print("Text:", comparison_text)
print("Simple word-level tokens:", word_level)
print("Word-level token count:", len(word_level))
print("tiktoken token count:", len(tiktoken_ids))
print("tiktoken pieces:", [repr(enc.decode_single_token_bytes(i)) for i in tiktoken_ids])


print("\n" + "=" * 72)
print("Part 4: Round-trip check")
print("=" * 72)

for text in examples:
    ids = enc.encode(text)
    decoded = enc.decode(ids)
    print(f"{decoded == text!s:5s} | {text!r}")


print("\n" + "=" * 72)
print("Key observations")
print("=" * 72)
print("1. A real tokenizer does not assume one token = one whole word.")
print("2. The tokenizer can split words into reusable subword/byte pieces.")
print("3. It can represent Chinese, code, URLs, numbers, and emoji without requiring a separate unknown-token path for each kind of text.")
print("4. Token IDs are the actual integers passed into an embedding layer.")
print("5. The exact tokenization depends on the selected encoding vocabulary and merge rules.")
print("6. This experiment uses cl100k_base for education; different modern models can use different tokenizer vocabularies.")

print("\nStep 24 complete.")
