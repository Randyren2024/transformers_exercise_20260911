import torch
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
# 1. Input
# ---------------------------------------------------------
# We use the same tiny context as the previous exercises.
tokens = ["I", "love"]
input_ids = torch.tensor([vocab[token] for token in tokens])

print("Input tokens:")
print(tokens)

print("\nInput token IDs:")
print(input_ids)

# ---------------------------------------------------------
# 2. Simple input embeddings
# ---------------------------------------------------------
# Fixed embeddings so that we can focus on understanding
# Multi-Head Attention rather than training an entire model.
embedding_matrix = torch.tensor([
    [2.0, 0.0, 0.0, 0.0],  # I
    [0.0, 2.0, 0.0, 0.0],  # love
    [0.0, 0.0, 2.0, 0.0],  # cats
], dtype=torch.float32)

x = embedding_matrix[input_ids]

print("\nInput embeddings:")
print(x)

print("\nInput shape:")
print(x.shape)

# ---------------------------------------------------------
# 3. Multi-Head Attention dimensions
# ---------------------------------------------------------
# Model dimension = 4.
# We split it into 2 heads.
# Each head gets 2 dimensions.
d_model = 4
num_heads = 2
head_dim = d_model // num_heads

print("\nModel dimension:", d_model)
print("Number of heads:", num_heads)
print("Head dimension:", head_dim)

# ---------------------------------------------------------
# 4. Hand-written Q/K/V projection matrices
# ---------------------------------------------------------
# Instead of hiding the split inside nn.MultiheadAttention,
# we make it visible.
#
# Head 1 mainly looks at the first two input dimensions.
# Head 2 mainly looks at the last two input dimensions.
W_q = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

W_k = W_q.clone()
W_v = W_q.clone()

Q = x @ W_q
K = x @ W_k
V = x @ W_v

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

# ---------------------------------------------------------
# 5. Split Q/K/V into heads
# ---------------------------------------------------------
# Before split:
#   shape = (seq_len, 4)
#
# After split:
#   shape = (num_heads, seq_len, head_dim)
Q_heads = Q.view(-1, num_heads, head_dim).transpose(0, 1)
K_heads = K.view(-1, num_heads, head_dim).transpose(0, 1)
V_heads = V.view(-1, num_heads, head_dim).transpose(0, 1)

print("\nQ split into heads:")
print(Q_heads)
print("Q heads shape:")
print(Q_heads.shape)

print("\nK split into heads:")
print(K_heads)
print("K heads shape:")
print(K_heads.shape)

print("\nV split into heads:")
print(V_heads)
print("V heads shape:")
print(V_heads.shape)

# ---------------------------------------------------------
# 6. Causal mask
# ---------------------------------------------------------
seq_len = x.shape[0]
mask = torch.tril(torch.ones(seq_len, seq_len))

print("\nCausal mask:")
print(mask)

# ---------------------------------------------------------
# 7. Attention for each head
# ---------------------------------------------------------
head_outputs = []
head_weights = []

for head in range(num_heads):
    Q_h = Q_heads[head]
    K_h = K_heads[head]
    V_h = V_heads[head]

    scores = Q_h @ K_h.T / math.sqrt(head_dim)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V_h

    head_weights.append(weights)
    head_outputs.append(output)

    print(f"\n--- Head {head + 1} ---")
    print("Q:")
    print(Q_h)
    print("K:")
    print(K_h)
    print("V:")
    print(V_h)
    print("Scaled scores after causal mask:")
    print(scores)
    print("Attention weights:")
    print(weights)
    print("Head output:")
    print(output)

# Stack heads:
# shape = (seq_len, num_heads * head_dim)
head_outputs_stacked = torch.cat(head_outputs, dim=-1)

print("\nConcatenated head outputs:")
print(head_outputs_stacked)

print("\nConcatenated output shape:")
print(head_outputs_stacked.shape)

# ---------------------------------------------------------
# 8. Final output projection W_o
# ---------------------------------------------------------
# In a real Transformer, concatenated heads are projected back
# into the model dimension with W_o.
W_o = torch.eye(d_model)

multi_head_output = head_outputs_stacked @ W_o

print("\nFinal Multi-Head Attention output:")
print(multi_head_output)

print("\nFinal output shape:")
print(multi_head_output.shape)

# ---------------------------------------------------------
# 9. Interpretation
# ---------------------------------------------------------
print("\nInterpretation:")
print("One attention head gives one view of the sequence.")
print("With 2 heads, the model computes 2 attention patterns in parallel.")
print("Each head produces a small representation of size head_dim = 2.")
print("The head outputs are concatenated back into d_model = 4.")
print("Then W_o mixes the heads back into the model representation.")
print()
print("The important shape transition is:")
print("  (seq_len, d_model)")
print("       -> split heads ->")
print("  (num_heads, seq_len, head_dim)")
print("       -> attention ->")
print("  (num_heads, seq_len, head_dim)")
print("       -> concatenate ->")
print("  (seq_len, d_model)")
