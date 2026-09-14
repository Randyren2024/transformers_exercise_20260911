import torch
import torch.nn as nn
import torch.optim as optim

# Vocabulary from the previous exercises.
vocab = {
    "I": 0,
    "love": 1,
    "cats": 2
}

id_to_token = {v: k for k, v in vocab.items()}

# Training example:
# Input context:  I love
# Target token:   cats
input_tokens = torch.tensor([[vocab["I"], vocab["love"]]])
target_token = torch.tensor([vocab["cats"]])

# Model settings.
vocab_size = len(vocab)
embedding_dim = 4
context_length = 2

# Embedding layer:
# Each token ID -> one learned 4-dimensional vector.
embedding = nn.Embedding(vocab_size, embedding_dim)

# We will combine the two token embeddings by concatenating them.
# 2 tokens x 4 dimensions = 8 input features.
linear = nn.Linear(context_length * embedding_dim, vocab_size)

# Cross-entropy loss compares the predicted logits with the correct token ID.
loss_function = nn.CrossEntropyLoss()

# Adam updates the embedding and linear-layer parameters.
optimizer = optim.Adam(
    list(embedding.parameters()) + list(linear.parameters()),
    lr=0.05
)

print("Input context:")
print("I love")

print("\nInput token IDs:")
print(input_tokens)

print("\nTarget token:")
print("cats")
print("Target token ID:")
print(target_token)

print("\nInitial embedding matrix:")
print(embedding.weight.data)

# Train the model.
for step in range(1, 501):
    # Look up both token embeddings.
    embeddings = embedding(input_tokens)

    # Shape: (batch_size, context_length, embedding_dim)
    # [1, 2, 4]
    print_once = step == 1
    if print_once:
        print("\nInput embeddings shape:")
        print(embeddings.shape)

    # Concatenate the two token embeddings into one 8-dimensional vector.
    context_vector = embeddings.view(1, -1)

    if print_once:
        print("\nContext vector shape:")
        print(context_vector.shape)
        print("Context vector:")
        print(context_vector)

    # Predict the next token.
    logits = linear(context_vector)

    # Compute the error.
    loss = loss_function(logits, target_token)

    # Clear old gradients.
    optimizer.zero_grad()

    # Calculate gradients.
    loss.backward()

    # Update learned parameters.
    optimizer.step()

    if step == 1 or step % 100 == 0:
        print(f"\nStep {step:3d} | Loss: {loss.item():.6f}")

print("\nLearned embedding matrix:")
print(embedding.weight.data)

print("\nLearned linear weights:")
print(linear.weight.data)

print("\nLearned linear bias:")
print(linear.bias.data)

# ---------------------------------------------------------
# Next-token prediction after training.
# ---------------------------------------------------------

with torch.no_grad():
    embeddings = embedding(input_tokens)
    context_vector = embeddings.view(1, -1)
    logits = linear(context_vector)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probabilities, dim=-1).item()
    predicted_token = id_to_token[predicted_token_id]

print("\nFinal input embeddings:")
print(embeddings)

print("\nFinal context vector:")
print(context_vector)

print("\nLogits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nPredicted next token ID:")
print(predicted_token_id)

print("\nPredicted next token:")
print(predicted_token)
