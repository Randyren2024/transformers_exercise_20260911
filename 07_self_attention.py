import torch
import math

# Vocabulary from the previous exercises
# I -> 0, love -> 1, cats -> 2
vocab = {
    "I": 0,
    "love": 1,
    "cats": 2
}

id_to_token = {v: k for k, v in vocab.items()}

# We reuse the idea of learned embeddings from the previous exercise.
# Here we use fixed example embeddings so that we can focus only on
# understanding Self-Attention.
embedding_matrix = torch.tensor([
    [2.0, 0.0, 0.0, 0.0],  # I
    [0.0, 2.0, 0.0, 0.0],  # love
    [0.0, 0.0, 2.0, 0.0]   # cats
], dtype=torch.float32)

# Input context: "I love"
input_token_ids = torch.tensor([0, 1])
input_embeddings = embedding_matrix[input_token_ids]

print("Input tokens:")
print([id_to_token[token_id.item()] for token_id in input_token_ids])

print("\nInput token IDs:")
print(input_token_ids)

print("\nInput embeddings:")
print(input_embeddings)

print("\nInput embeddings shape:")
print(input_embeddings.shape)

# ---------------------------------------------------------
# 1. Create Query (Q), Key (K), and Value (V)
# ---------------------------------------------------------
# For this first experiment, we use simple hand-written matrices.
# They are intentionally easy to inspect.
W_q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0]
])

W_k = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0]
])

W_v = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0]
])

Q = input_embeddings @ W_q
K = input_embeddings @ W_k
V = input_embeddings @ W_v

print("\nQ (Query):")
print(Q)

print("\nK (Key):")
print(K)

print("\nV (Value):")
print(V)

# ---------------------------------------------------------
# 2. Calculate attention scores: Q @ K^T
# ---------------------------------------------------------
attention_scores = Q @ K.T

print("\nRaw attention scores (Q @ K^T):")
print(attention_scores)

# ---------------------------------------------------------
# 3. Scale the scores
# ---------------------------------------------------------
# Standard scaled dot-product attention divides by sqrt(d_k).
d_k = K.shape[-1]
scaled_scores = attention_scores / math.sqrt(d_k)

print("\nScaled attention scores:")
print(scaled_scores)

# ---------------------------------------------------------
# 4. Softmax -> attention weights
# ---------------------------------------------------------
attention_weights = torch.softmax(scaled_scores, dim=-1)

print("\nAttention weights:")
print(attention_weights)

print("\nAttention weights row sums:")
print(attention_weights.sum(dim=-1))

# ---------------------------------------------------------
# 5. Weighted sum of Values
# ---------------------------------------------------------
context = attention_weights @ V

print("\nAttention output (weighted values):")
print(context)

print("\nAttention output shape:")
print(context.shape)

# ---------------------------------------------------------
# 6. Focus on the second token: "love"
# ---------------------------------------------------------
print("\nRepresentation for the token 'love' after attention:")
print(context[1])

print("\nInterpretation:")
print("The second row shows the new representation of 'love' after it")
print("looked at both 'I' and 'love'. The attention weights tell us how")
print("much it attended to each token.")
