import numpy as np

# Vocabulary from 01_tokenization.py
# I -> 0, love -> 1, cats -> 2
vocab = {
    "I": 0,
    "love": 1,
    "cats": 2
}

id_to_token = {v: k for k, v in vocab.items()}

# Embedding matrix from 02_embedding.py
embedding_matrix = np.array([
    [0.2, 0.1, 0.7, 0.3],  # token 0: I
    [0.8, 0.2, 0.4, 0.1],  # token 1: love
    [0.3, 0.9, 0.1, 0.6]   # token 2: cats
])

# Our input is "I love".
# For this first simple experiment, we use the LAST token: "love".
input_token_id = vocab["love"]
input_embedding = embedding_matrix[input_token_id]

# Output weight matrix.
# Shape: embedding_dim x vocab_size = 4 x 3
# These are hand-written weights for demonstration.
# They are NOT learned yet.
output_weights = np.array([
    [0.1, 0.0, 0.2],
    [0.0, 0.1, 1.0],
    [0.2, 0.1, 0.0],
    [0.0, 0.2, 0.8]
])

# Bias for each vocabulary token.
output_bias = np.array([0.0, 0.0, 0.0])

# Linear layer: embedding -> logits
logits = input_embedding @ output_weights + output_bias

# Softmax: logits -> probabilities
exp_logits = np.exp(logits - np.max(logits))
probabilities = exp_logits / np.sum(exp_logits)

# Choose the token with the highest probability.
predicted_token_id = np.argmax(probabilities)
predicted_token = id_to_token[predicted_token_id]

print("Input token:")
print("love")

print("\nInput token ID:")
print(input_token_id)

print("\nInput embedding:")
print(input_embedding)

print("\nInput embedding shape:")
print(input_embedding.shape)

print("\nLogits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nPredicted next token ID:")
print(predicted_token_id)

print("\nPredicted next token:")
print(predicted_token)
