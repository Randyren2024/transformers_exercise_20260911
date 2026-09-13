import numpy as np

# Token IDs from the tokenizer
# I -> 0, love -> 1, cats -> 2
token_ids = [0, 1, 2]

# Embedding matrix
# 3 rows = vocabulary size
# 4 columns = embedding dimension
embedding_matrix = np.array([
    [0.2, 0.1, 0.7, 0.3],  # token 0: I
    [0.8, 0.2, 0.4, 0.1],  # token 1: love
    [0.3, 0.9, 0.1, 0.6]   # token 2: cats
])

# Look up the embedding vector for each token ID
embeddings = embedding_matrix[token_ids]

print("Embedding Matrix:")
print(embedding_matrix)

print("\nEmbedding Matrix Shape:")
print(embedding_matrix.shape)

print("\nToken IDs:")
print(token_ids)

print("\nEmbeddings:")
print(embeddings)

print("\nEmbeddings Shape:")
print(embeddings.shape)
