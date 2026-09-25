import os
d = "/content/drive/MyDrive/transformers_exercise_20260911/data/step58_v2_corpus"
print("Files:", os.listdir(d))
import numpy as np
for name in os.listdir(d):
    f = os.path.join(d, name)
    sz = os.path.getsize(f)
    if name.endswith(".uint16"):
        n_tokens = sz // 2
        print(f"{name}: {sz} bytes = {n_tokens} tokens = {n_tokens/2e8:.3f}B tokens")
    else:
        print(f"{name}: {sz} bytes")
