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
# At each position, predict the token that comes NEXT:
#   position 0: "I"    -> "love"
#   position 1: "love" -> "cats"
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
seq_len = len(inputs)
embedding_dim = 4

# Token embedding: what the token means.
token_embedding = nn.Embedding(vocab_size, embedding_dim)

# Positional embedding: where the token is in the sequence.
# Position 0 gets one learnable vector.
# Position 1 gets another learnable vector.
position_embedding = nn.Embedding(seq_len, embedding_dim)

# Learnable attention projections.
W_q = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_k = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_v = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
W_o = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)

# Attention output -> vocabulary logits.
linear = nn.Linear(embedding_dim, vocab_size)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    [
        token_embedding.weight,
        position_embedding.weight,
        W_q,
        W_k,
        W_v,
        W_o,
    ] + list(linear.parameters()),
    lr=0.05,
)

print("\nInitial token embedding matrix:")
print(token_embedding.weight.data)

print("\nInitial positional embedding matrix:")
print(position_embedding.weight.data)

# ---------------------------------------------------------
# 3. Positional information
# ---------------------------------------------------------
position_ids = torch.arange(seq_len)

print("\nPosition IDs:")
print(position_ids)

initial_token_embeddings = token_embedding(inputs)
initial_position_embeddings = position_embedding(position_ids)

print("\nToken embeddings:")
print(initial_token_embeddings)

print("\nPositional embeddings:")
print(initial_position_embeddings)

# The Transformer input is token information + position information.
x_with_position = initial_token_embeddings + initial_position_embeddings

print("\nToken + positional embeddings:")
print(x_with_position)

print("\nShape:")
print(x_with_position.shape)

# ---------------------------------------------------------
# 4. Causal mask
# ---------------------------------------------------------
mask = torch.tril(torch.ones(seq_len, seq_len))

print("\nCausal mask (1 = can see, 0 = blocked):")
print(mask)

# ---------------------------------------------------------
# 5. Training
# ---------------------------------------------------------
for step in range(1, 501):
    # Token embedding + positional embedding
    token_vectors = token_embedding(inputs)
    position_vectors = position_embedding(position_ids)
    x = token_vectors + position_vectors

    # Self-attention
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    scores = Q @ K.T / math.sqrt(embedding_dim)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    context = attention_weights @ V

    # Output projection
    attention_output = context @ W_o

    # Predict next token at every position
    logits = linear(attention_output)
    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

# ---------------------------------------------------------
# 6. Inspect learned token and position information
# ---------------------------------------------------------
print("\nLearned token embedding matrix:")
print(token_embedding.weight.detach())

print("\nLearned positional embedding matrix:")
print(position_embedding.weight.detach())

# ---------------------------------------------------------
# 7. Final prediction
# ---------------------------------------------------------
with torch.no_grad():
    token_vectors = token_embedding(inputs)
    position_vectors = position_embedding(position_ids)
    x = token_vectors + position_vectors

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

print("\nFinal token embeddings:")
print(token_vectors)

print("\nFinal positional embeddings:")
print(position_vectors)

print("\nFinal model input (token + position):")
print(x)

print("\nAttention weights:")
print(attention_weights)

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
# 8. Interpretation
# ---------------------------------------------------------
print("\nInterpretation:")
print("Token embedding answers: WHAT token is this?")
print("Positional embedding answers: WHERE is this token?")
print("\nThe model combines them by addition:")
print("    model input = token embedding + positional embedding")
print("\nThis lets the same token have different representations at")
print("different positions in a sequence. The positional vectors")
print("are learned parameters, just like token embeddings and the")
print("attention weights.")
