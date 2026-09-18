# Node 54 — Model Freeze and Deployment Baseline

## Model snapshot

- Parameters: 8,413,696
- BPE vocabulary: 8,000
- Transformer blocks: 8
- Hidden size: 256
- Attention heads: 8
- FFN size: 1,024
- Context: 256 tokens
- Input/output embeddings: tied
- Language focus: English + Chinese

Deployment candidate:
artifacts/step54/tiny_gpt_step54_best.pt

Tokenizer:
artifacts/step43/step43_bpe_8000.json

## What the experiments established

Step 43 and Step 45 established the bilingual pretrained base.
Steps 47–54 progressively added curated instruction tuning and precision tuning.
Step 52 separated knowledge from decoding and showed that the model can learn some exact answers while still preferring wrong answers for several arithmetic questions.
Step 53 improved short bilingual mappings and simple instruction behavior, but arithmetic remained unreliable.
Step 54 added a contrastive ranking objective. It improved some answer ranking during training, but the final probes still show that arithmetic and some English instruction responses are unreliable.

Therefore Step 54 is frozen as a deployment baseline and learning milestone, not as a production-quality general assistant.

## Why deployment is still useful

The model is now small enough to run as a normal application model.
The service does not need the Hugging Face Transformers runtime. It only needs the exact TinyGPT architecture, PyTorch, the tokenizer, and the checkpoint.
The model can be loaded once when the Flask application starts and reused for requests.

## Target architecture

Browser
  -> Flask web application
  -> /api/chat
  -> TinyGPTService
  -> PyTorch TinyGPT
  -> 8K BPE tokenizer
  -> response

Keep the API contract independent from the checkpoint so later training experiments can replace the .pt file without rebuilding the frontend.

## Node 55

Local Flask deployment of the frozen Step 54 checkpoint.

First prove:
1. The checkpoint loads outside Colab.
2. The tokenizer matches.
3. /api/chat returns a generated answer.
4. A browser can send a message and display it.