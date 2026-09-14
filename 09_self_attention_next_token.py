import torch
import torch.nn as nn
import torch.optim as optim
import math

# Reproducibility
torch.manual_seed(42)

# Vocabulary from the previous exercises
# I -> 0, love -> 1, cats -> 2
vocab = {
    "I": 0,
    "love": 1,
    "cats": 2
}

id_to_token = {v: k for k, v in vocab.items()}

# ---------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------
# Same setup as 05_multiple_examples.py.
# Each position predicts the token that comes NEXT:
#   position 0: "I"    -> predict "love"
#   position 1: "love" -> predict "cats"
# This is exactly how GPT is trained: predict every next token,
# at every position, in parallel.
inputs = torch.tensor([0, 1])   # "I love"
targets = torch.tensor([1, 2])  # "love cats"

print("Training inputs:")
print([id_to_token[i.item()] for i in inputs])

print("\nTraining targets:")
print([id_to_token[i.item()] for i in targets])

# ---------------------------------------------------------
# 2. Model: Embedding + Causal Self-Attention + Linear
# ---------------------------------------------------------
# This is the smallest possible GPT-style block:
#
#   token IDs
#       |
#   Embedding
#       |
#   Q, K, V projections
#       |
#   Scaled dot-product attention
#       |
#   Causal mask  (no peeking at the future)
#       |
#   Softmax -> attention weights
#       |
#   Weighted sum of V
#       |
#   Output projection W_o
#       |
#   Linear -> logits
#       |
#   next-token prediction
#
vocab_size = len(vocab)
embedding_dim = 4

# Learnable token embeddings (same idea as 04/05/06).
embedding = nn.Embedding(vocab_size, embedding_dim)

# In 07/08 we used hand-written identity matrices for W_q/W_k/W_v.
# Here they are LEARNABLE so the model can actually adjust how
# each token should attend to others.
W_q = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_k = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_v = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_o = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)

# Final linear layer: attention output -> vocab logits.
linear = nn.Linear(embedding_dim, vocab_size)

# Cross-entropy loss compares predicted logits with the correct next token.
loss_function = nn.CrossEntropyLoss()

# Adam updates ALL learned parameters together.
optimizer = optim.Adam(
    [embedding.weight, W_q, W_k, W_v, W_o] + list(linear.parameters()),
    lr=0.05
)

print("\nInitial embedding matrix:")
print(embedding.weight.data)

# ---------------------------------------------------------
# 3. Causal mask (same idea as 08_causal_mask.py)
# ---------------------------------------------------------
seq_len = inputs.shape[0]
mask = torch.tril(torch.ones(seq_len, seq_len))

print("\nCausal mask (1 = can see, 0 = blocked):")
print(mask)

# ---------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------
for step in range(1, 501):
    # 1) Embedding lookup
    # x shape: (seq_len, embedding_dim)
    x = embedding(inputs)

    # 2) Project into Q, K, V
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    # 3) Scaled dot-product attention
    scores = Q @ K.T / math.sqrt(embedding_dim)

    # 4) Apply causal mask (block the future)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    # 5) Softmax -> attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # 6) Weighted sum of Values
    context = attention_weights @ V

    # 7) Output projection
    attention_output = context @ W_o

    # 8) Linear layer -> logits (one row per position)
    logits = linear(attention_output)

    # 9) Loss: every position predicts the next token
    loss = loss_function(logits, targets)

    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

# ---------------------------------------------------------
# 5. Inspect what was learned
# ---------------------------------------------------------
print("\nLearned embedding matrix:")
print(embedding.weight.detach())

# ---------------------------------------------------------
# 6. Next-token prediction after training
# ---------------------------------------------------------
with torch.no_grad():
    x = embedding(inputs)
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    scores = Q @ K.T / math.sqrt(embedding_dim)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    context = attention_weights @ V
    attention_output = context @ W_o
    logits = linear(attention_output)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(probabilities, dim=-1)

print("\nAttention weights:")
print(attention_weights)

print("\nLogits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nPredicted token IDs:")
print(predicted_ids)

print("\nPredicted tokens (one per position):")
print([id_to_token[i.item()] for i in predicted_ids])

print("\nExpected tokens (targets):")
print([id_to_token[i.item()] for i in targets])

# ---------------------------------------------------------
# 7. Interpretation
# ---------------------------------------------------------
print("\nInterpretation:")
print("This is the smallest possible GPT-style block:")
print("  Embedding -> Causal Self-Attention -> Linear -> next token")
print()
print("Each position predicts the token that should come NEXT:")
print("  position 0 ('I')    -> should predict 'love'")
print("  position 1 ('love') -> should predict 'cats'")
print()
print("The Causal Mask is what makes this GPT-style training fair:")
print("when 'I' is predicting 'love', it cannot peek at the 'love'")
print("embedding sitting in position 1. It can only attend to itself")
print("and to tokens that come BEFORE it.")
print()
print("Compare this to 06_context_prediction.py, which concatenated")
print("the two embeddings into one long vector and only predicted ONE")
print("token. Here we predict a next token at EVERY position, in")
print("parallel, using the same attention block - exactly like a")
print("real GPT does during training.")
