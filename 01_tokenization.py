# 01_tokenization.py

# 1. 原始文本
text = "I love cats"

# 2. 把句子切成 token
tokens = text.split()

# 3. 建立 vocabulary
vocab = {}

for token in tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

# 4. token → ID
token_ids = []

for token in tokens:
    token_ids.append(vocab[token])

# 5. ID → token
id_to_token = {}

for token, token_id in vocab.items():
    id_to_token[token_id] = token

decoded_tokens = []

for token_id in token_ids:
    decoded_tokens.append(id_to_token[token_id])


# 6. 打印结果
print("Original text:", text)
print("Tokens:", tokens)
print("Vocabulary:", vocab)
print("Token IDs:", token_ids)
print("Decoded tokens:", decoded_tokens)