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
# 1. Training data
# ---------------------------------------------------------
# Each position predicts the token that comes NEXT:
# I    -> love
# love -> cats
inputs = torch.tensor([0, 1])
targets = torch.tensor([1, 2])

print("Training inputs:")
print([id_to_token[i.item()] for i in inputs])

print("\nTraining targets:")
print([id_to_token[i.item()] for i in targets])

# ---------------------------------------------------------
# 2. Model dimensions
# ---------------------------------------------------------
vocab_size = len(vocab)
embedding_dim = 4
hidden_dim = 8
seq_len = inputs.shape[0]

# ---------------------------------------------------------
# 3. Token embedding + positional embedding
# ---------------------------------------------------------
token_embedding = nn.Embedding(vocab_size, embedding_dim)
position_embedding = nn.Embedding(seq_len, embedding_dim)

position_ids = torch.arange(seq_len)

# ---------------------------------------------------------
# 4. Learnable self-attention weights
# ---------------------------------------------------------
W_q = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_k = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_v = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_o = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)

# ---------------------------------------------------------
# 5. Transformer Block components
# ---------------------------------------------------------
# Feed-Forward Network (FFN):
# 4 -> 8 -> 4
# GELU is commonly used in Transformer feed-forward layers.
ffn = nn.Sequential(
    nn.Linear(embedding_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, embedding_dim),
)

# LayerNorm stabilizes the hidden representations.
layer_norm_1 = nn.LayerNorm(embedding_dim)
layer_norm_2 = nn.LayerNorm(embedding_dim)

# Final language-model head: hidden state -> vocabulary logits.
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
# 6. Causal mask
# ---------------------------------------------------------
mask = torch.tril(torch.ones(seq_len, seq_len))

print("\nCausal mask:")
print(mask)

# ---------------------------------------------------------
# 7. Forward pass: one Transformer Block
# ---------------------------------------------------------
def forward():
    # Token + position
    x = token_embedding(inputs) + position_embedding(position_ids)

    # ---------------- Self-Attention ----------------
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    scores = Q @ K.T / math.sqrt(embedding_dim)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)

    attention_output = attention_weights @ V
    attention_output = attention_output @ W_o

    # Residual connection 1 + LayerNorm
    x = layer_norm_1(x + attention_output)

    # ---------------- Feed-Forward Network ----------------
    ffn_output = ffn(x)

    # Residual connection 2 + LayerNorm
    x = layer_norm_2(x + ffn_output)

    # ---------------- Language-model head ----------------
    logits = lm_head(x)

    return logits, attention_weights, x

# ---------------------------------------------------------
# 8. Training loop
# ---------------------------------------------------------
for step in range(1, 501):
    logits, attention_weights, hidden_states = forward()

    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

# ---------------------------------------------------------
# 9. Inspect the learned result
# ---------------------------------------------------------
with torch.no_grad():
    logits, attention_weights, hidden_states = forward()
    probabilities = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(probabilities, dim=-1)

print("\nFinal attention weights:")
print(attention_weights)

print("\nFinal hidden states after Transformer Block:")
print(hidden_states)

print("\nLogits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nPredicted token IDs:")
print(predicted_ids)

print("\nPredicted tokens:")
print([id_to_token[i.item()] for i in predicted_ids])

print("\nExpected tokens:")
print([id_to_token[i.item()] for i in targets])

# ---------------------------------------------------------
# 10. Print the Transformer Block pipeline
# ---------------------------------------------------------
print("\nTransformer Block pipeline:")
print("Token Embedding + Position Embedding")
print("        ↓")
print("Causal Self-Attention")
print("        ↓")
print("Residual Connection + LayerNorm")
print("        ↓")
print("Feed-Forward Network (FFN)")
print("        ↓")
print("Residual Connection + LayerNorm")
print("        ↓")
print("LM Head")
print("        ↓")
print("Next-token prediction")

print("\nInterpretation:")
print("Self-Attention lets each token mix information from earlier tokens.")
print("The first residual connection keeps the original representation")
print("while adding the attention result.")
print("The Feed-Forward Network then transforms each token representation")
print("independently, using the same small neural network at every position.")
print("The second residual connection adds the FFN result back, and LayerNorm")
print("stabilizes the representation before the final language-model head.")
