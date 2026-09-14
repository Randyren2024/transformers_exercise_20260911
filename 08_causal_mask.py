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

# We reuse the fixed example embeddings from 07_self_attention.py
# so that the ONLY new idea in this file is the Causal Mask.
embedding_matrix = torch.tensor([
    [2.0, 0.0, 0.0, 0.0],  # I
    [0.0, 2.0, 0.0, 0.0],  # love
    [0.0, 0.0, 2.0, 0.0]   # cats
], dtype=torch.float32)

# Input context: "I love"
# We keep 2 tokens, exactly like 07_self_attention.py, so the only
# thing that changes here is the mask.
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
# Same hand-written identity-style weights as in 07_self_attention.py.
# Q = K = V on purpose, so the only thing that changes the output
# in this file is the Causal Mask itself.
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
d_k = K.shape[-1]
scaled_scores = attention_scores / math.sqrt(d_k)

print("\nScaled attention scores (BEFORE mask):")
print(scaled_scores)

# ---------------------------------------------------------
# 4. Build the Causal Mask   <-- the only new idea in this file
# ---------------------------------------------------------
# Without a mask, every token can attend to every other token,
# including tokens that come LATER in the sequence. That is fine
# for understanding context, but during training for next-token
# prediction it lets the model "cheat" by looking at the answer.
#
# A Causal Mask forces each token to only look at itself and the
# tokens BEFORE it. The "future" positions are blocked.
#
# We use a lower-triangular mask of shape (seq_len, seq_len):
#   position i is allowed to attend to position j   iff   j <= i
#
# Allowed positions   -> keep the score
# Disallowed (future)  -> set the score to -inf so that after
#                         Softmax they become exactly 0.

seq_len = input_embeddings.shape[0]

# torch.tril gives 1 on and below the diagonal, 0 strictly above.
mask = torch.tril(torch.ones(seq_len, seq_len))

print("\nCausal mask (1 = can see, 0 = blocked):")
print(mask)

# masked_fill replaces positions where mask == 0 with -inf.
masked_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

print("\nScaled attention scores AFTER mask (-inf = blocked):")
print(masked_scores)

# ---------------------------------------------------------
# 5. Softmax -> attention weights
# ---------------------------------------------------------
# Because masked positions are -inf, exp(-inf) = 0, so the blocked
# positions become exactly 0 in the weights.
attention_weights = torch.softmax(masked_scores, dim=-1)

print("\nAttention weights (after causal mask):")
print(attention_weights)

print("\nAttention weights row sums:")
print(attention_weights.sum(dim=-1))

# ---------------------------------------------------------
# 6. Weighted sum of Values
# ---------------------------------------------------------
context = attention_weights @ V

print("\nAttention output (weighted values):")
print(context)

print("\nAttention output shape:")
print(context.shape)

# ---------------------------------------------------------
# 7. Compare: with mask vs. without mask
# ---------------------------------------------------------
# Recompute the unmasked version (this is exactly what
# 07_self_attention.py did) so the difference is visible side by side.
unmasked_weights = torch.softmax(scaled_scores, dim=-1)
unmasked_context = unmasked_weights @ V

print("\n--- WITHOUT mask (same as 07_self_attention.py) ---")
print("Attention weights:")
print(unmasked_weights)
print("Context output:")
print(unmasked_context)

print("\n--- WITH causal mask (this file) ---")
print("Attention weights:")
print(attention_weights)
print("Context output:")
print(context)

# ---------------------------------------------------------
# 8. Interpretation
# ---------------------------------------------------------
print("\nInterpretation:")
print("Row 0 (token 'I') can ONLY see itself, because there is no")
print("previous token. Its new representation is therefore exactly")
print("its own Value vector, with no mixing from 'love'.")
print()
print("Row 1 (token 'love') can see 'I' and itself, exactly like in")
print("07_self_attention.py, because 'I' is in the past. The mask")
print("only blocks the FUTURE, and 'I' is not in the future of 'love'.")
print()
print("This is exactly what a GPT model needs: when predicting the")
print("token that comes AFTER 'love', the model must not peek at that")
print("future token during training. The Causal Mask enforces this.")
