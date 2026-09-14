import torch
import torch.nn as nn
import torch.optim as optim
import math

# Reproducibility
torch.manual_seed(42)

# Vocabulary
# I -> 0, love -> 1, cats -> 2
vocab = {
    "I": 0,
    "love": 1,
    "cats": 2,
}

id_to_token = {v: k for k, v in vocab.items()}

# ---------------------------------------------------------
# 1. Training data (same as 10 and 11)
# ---------------------------------------------------------
# Each position predicts the token that comes NEXT:
# I    -> love
# love -> cats
inputs = torch.tensor([0, 1])
targets = torch.tensor([1, 2])

print("Training data:")
print("  inputs:  ['I', 'love']")
print("  targets: ['love', 'cats']")

# ---------------------------------------------------------
# 2. Model dimensions
# ---------------------------------------------------------
vocab_size = len(vocab)
embedding_dim = 4
hidden_dim = 8

# IMPORTANT CHANGE vs 11_transformer_block.py:
# The position embedding table must be sized for the GENERATION
# length, not the training length. Training only ever uses
# positions 0 and 1, but during generation the sequence grows:
# 1 token, 2 tokens, 3 tokens, ...
# If this table only had 2 rows, generation would crash the
# moment the context grew past 2 tokens.
max_seq_len = 5
position_embedding = nn.Embedding(max_seq_len, embedding_dim)

token_embedding = nn.Embedding(vocab_size, embedding_dim)

# ---------------------------------------------------------
# 3. Transformer Block components (same as 11)
# ---------------------------------------------------------
W_q = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_k = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_v = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_o = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)

ffn = nn.Sequential(
    nn.Linear(embedding_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, embedding_dim),
)

layer_norm_1 = nn.LayerNorm(embedding_dim)
layer_norm_2 = nn.LayerNorm(embedding_dim)

lm_head = nn.Linear(embedding_dim, vocab_size)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    [
        token_embedding.weight,
        position_embedding.weight,
        W_q,
        W_k,
        W_v,
        W_o,
        *ffn.parameters(),
        *layer_norm_1.parameters(),
        *layer_norm_2.parameters(),
        *lm_head.parameters(),
    ],
    lr=0.03,
)

# ---------------------------------------------------------
# 4. Forward pass for ANY length of context
# ---------------------------------------------------------
# In 11 the sequence length was fixed at 2, so the mask and
# positions could be built once outside the loop.
# For generation, the context grows every step, so the forward
# pass must build positions and the causal mask dynamically.
def forward(token_ids):
    seq_len = token_ids.shape[0]

    # Token + position
    x = token_embedding(token_ids) + position_embedding(torch.arange(seq_len))

    # Causal mask for THIS length
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # Causal self-attention
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    scores = Q @ K.T / math.sqrt(embedding_dim)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)

    attention_output = attention_weights @ V
    attention_output = attention_output @ W_o

    # Residual + LayerNorm
    x = layer_norm_1(x + attention_output)

    # FFN + Residual + LayerNorm
    x = layer_norm_2(x + ffn(x))

    logits = lm_head(x)
    return logits

# ---------------------------------------------------------
# 5. Training loop (same as 11)
# ---------------------------------------------------------
print("\nTraining...")
for step in range(1, 501):
    logits = forward(inputs)
    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

# ---------------------------------------------------------
# 6. Autoregressive generation
# ---------------------------------------------------------
# This is the loop that makes GPT feel like it is "writing":
#
#   start with a prompt:            [I]
#   predict the next token:         love
#   append it and repeat:           [I, love]
#   predict the next token:         cats
#   append it and repeat:           [I, love, cats]
#   ...
#
# Each step re-runs the WHOLE Transformer Block on the whole
# context, but we only use the LAST position's logits, because
# the last position is the one predicting the NEXT token.
print("\n--- Generation (greedy: always pick the highest probability) ---")

max_new_tokens = 4
generated_ids = [vocab["I"]]

for step in range(1, max_new_tokens + 1):
    context_ids = torch.tensor(generated_ids)

    with torch.no_grad():
        logits = forward(context_ids)

    # Only the LAST position predicts the next token.
    last_logits = logits[-1]
    probabilities = torch.softmax(last_logits, dim=-1)
    next_id = torch.argmax(probabilities).item()

    context_tokens = [id_to_token[i] for i in generated_ids]

    print(f"\nStep {step}:")
    print(f"  Context: {context_tokens}")
    print("  Probabilities for the next token:")
    for token_id in range(vocab_size):
        token = id_to_token[token_id]
        print(f"    {token:5s}: {probabilities[token_id].item():.6f}")
    print(f"  Chosen: '{id_to_token[next_id]}'")

    generated_ids.append(next_id)

print("\nGenerated sentence:")
print(" ".join(id_to_token[i] for i in generated_ids))

# ---------------------------------------------------------
# 7. Second generation from a different prompt
# ---------------------------------------------------------
print("\n--- Generation from a different prompt: 'love' ---")

generated_ids_2 = [vocab["love"]]

for step in range(1, 3):
    context_ids = torch.tensor(generated_ids_2)

    with torch.no_grad():
        logits = forward(context_ids)

    last_logits = logits[-1]
    probabilities = torch.softmax(last_logits, dim=-1)
    next_id = torch.argmax(probabilities).item()

    context_tokens = [id_to_token[i] for i in generated_ids_2]
    print(f"  Step {step}: context {context_tokens} -> '{id_to_token[next_id]}'")

    generated_ids_2.append(next_id)

print("\nGenerated sentence:")
print(" ".join(id_to_token[i] for i in generated_ids_2))

# ---------------------------------------------------------
# 8. Interpretation
# ---------------------------------------------------------
print("\nInterpretation:")
print("Generation is AUTOREGRESSIVE: the model's own output is fed")
print("back in as input, one token at a time. That is the 'G' in GPT:")
print("Generative.")
print()
print("At every step we only read logits[-1], the LAST position,")
print("because that position has seen the whole context (thanks to")
print("the causal mask) and is the one that predicts the NEXT token.")
print()
print("The position embedding table is sized max_seq_len=5, bigger")
print("than the training length of 2, so the context can grow during")
print("generation. Rows for positions 2, 3, 4 were never trained.")
print()
print("Notice what happens after the context becomes 'I love cats':")
print("the probabilities stop being confident. During training the")
print("model NEVER saw 'cats' as a context token (it was only ever a")
print("target), so past that point it is extrapolating. Real GPTs are")
print("trained on billions of tokens so almost every context is")
print("familiar, and they also sample with temperature / top-k / top-p")
print("instead of always taking the single most likely token.")
