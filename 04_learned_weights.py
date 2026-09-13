import torch
import torch.nn as nn

# Make the experiment reproducible.
torch.manual_seed(42)

# Our tiny vocabulary from 01_tokenization.py
vocab = {
    "I": 0,
    "love": 1,
    "cats": 2
}

id_to_token = {v: k for k, v in vocab.items()}

# ------------------------------------------------------------
# 1. Training data
# ------------------------------------------------------------
# We will teach the model one simple rule:
# "love" -> "cats"

input_token_id = torch.tensor([vocab["love"]])
target_token_id = torch.tensor([vocab["cats"]])

# ------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------
# Embedding: token ID -> 4-dimensional vector
embedding = nn.Embedding(num_embeddings=3, embedding_dim=4)

# Linear layer: 4-dimensional vector -> 3 logits
linear = nn.Linear(in_features=4, out_features=3)

# These weights are RANDOMLY initialized.
# We are NOT writing the weights by hand anymore.

print("Initial embedding matrix:")
print(embedding.weight.detach())

print("\nInitial linear weights:")
print(linear.weight.detach())

print("\nInitial linear bias:")
print(linear.bias.detach())

# ------------------------------------------------------------
# 3. Loss function and optimizer
# ------------------------------------------------------------
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    list(embedding.parameters()) + list(linear.parameters()),
    lr=0.1
)

# ------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------
for step in range(200):
    # Forward pass
    input_embedding = embedding(input_token_id)
    logits = linear(input_embedding)

    # Compare prediction with the correct answer: "cats"
    loss = loss_function(logits, target_token_id)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update the learned parameters
    optimizer.step()

    if step == 0 or (step + 1) % 20 == 0:
        print(f"Step {step + 1:3d} | Loss: {loss.item():.6f}")

# ------------------------------------------------------------
# 5. Show what was learned
# ------------------------------------------------------------
print("\nLearned embedding matrix:")
print(embedding.weight.detach())

print("\nLearned linear weights:")
print(linear.weight.detach())

print("\nLearned linear bias:")
print(linear.bias.detach())

# ------------------------------------------------------------
# 6. Next-token prediction
# ------------------------------------------------------------
with torch.no_grad():
    input_embedding = embedding(input_token_id)
    logits = linear(input_embedding)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probabilities, dim=-1).item()
    predicted_token = id_to_token[predicted_token_id]

print("\nInput token:")
print("love")

print("\nInput token ID:")
print(input_token_id.item())

print("\nInput embedding:")
print(input_embedding.squeeze(0))

print("\nLogits:")
print(logits.squeeze(0))

print("\nProbabilities:")
print(probabilities.squeeze(0))

print("\nPredicted next token ID:")
print(predicted_token_id)

print("\nPredicted next token:")
print(predicted_token)
