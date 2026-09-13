import torch
import torch.nn as nn

# Reproducibility
# This makes the random initialization the same each time we run the script.
torch.manual_seed(42)

# Vocabulary from 01_tokenization.py
# I -> 0, love -> 1, cats -> 2
vocab = {
    "I": 0,
    "love": 1,
    "cats": 2
}

id_to_token = {v: k for k, v in vocab.items()}

# ---------------------------------------------------------
# Training data
# ---------------------------------------------------------
# We now have TWO training examples:
# I    -> love
# love -> cats
#
# Input token IDs
inputs = torch.tensor([0, 1])

# Correct next-token IDs (targets)
targets = torch.tensor([1, 2])

print("Training inputs:")
print(inputs)

print("\nTraining targets:")
print(targets)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
# Each token gets a 4-dimensional embedding.
embedding = nn.Embedding(num_embeddings=3, embedding_dim=4)

# Convert a 4-dimensional embedding into 3 logits,
# one score for each token in the vocabulary.
linear = nn.Linear(in_features=4, out_features=3)

# CrossEntropyLoss combines softmax + negative log likelihood.
loss_function = nn.CrossEntropyLoss()

# Adam updates the learned parameters.
optimizer = torch.optim.Adam(
    list(embedding.parameters()) + list(linear.parameters()),
    lr=0.05
)

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
for step in range(1, 501):
    # 1. Embedding lookup
    input_embeddings = embedding(inputs)

    # 2. Linear layer -> logits
    logits = linear(input_embeddings)

    # 3. Compare predictions with the correct next tokens
    loss = loss_function(logits, targets)

    # 4. Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # 5. Update embedding and linear weights
    optimizer.step()

    if step == 1 or step % 100 == 0:
        print(f"\nStep {step:3d} | Loss: {loss.item():.6f}")

# ---------------------------------------------------------
# Inspect learned parameters
# ---------------------------------------------------------
print("\nLearned embedding matrix:")
print(embedding.weight.detach())

print("\nLearned linear weights:")
print(linear.weight.detach())

print("\nLearned linear bias:")
print(linear.bias.detach())

# ---------------------------------------------------------
# Next-token prediction
# ---------------------------------------------------------
def predict_next_token(token):
    token_id = vocab[token]

    with torch.no_grad():
        # Token -> learned embedding
        input_embedding = embedding(torch.tensor([token_id]))

        # Learned embedding -> logits
        logits = linear(input_embedding)

        # Logits -> probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Highest probability token
        predicted_token_id = torch.argmax(probabilities, dim=-1).item()

    print(f"\nInput token: {token}")
    print(f"Input token ID: {token_id}")
    print("Input embedding:")
    print(input_embedding.squeeze(0))
    print("Logits:")
    print(logits.squeeze(0))
    print("Probabilities:")
    print(probabilities.squeeze(0))
    print(f"Predicted next token ID: {predicted_token_id}")
    print(f"Predicted next token: {id_to_token[predicted_token_id]}")


# Test both learned relationships.
predict_next_token("I")
predict_next_token("love")
