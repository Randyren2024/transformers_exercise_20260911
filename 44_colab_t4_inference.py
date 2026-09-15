import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Step 44: Load the Step 43B checkpoint and test real generation.
# Everything lives on Google Drive in the T4 session.

ROOT = Path("/content/drive/MyDrive/transformers_exercise_20260911")
CHECKPOINT = ROOT / "artifacts" / "step43" / "tiny_gpt_step43b.pt"
TOKENIZER = ROOT / "artifacts" / "step43" / "step43_bpe_8000.json"

VOCAB_SIZE = 8000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.head_dim = head_dim

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        k = k.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        v = v.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, D_MODEL)
        return self.out(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, bias=False),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        t = x.size(1)
        p = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(p)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h))


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.8, top_k=40):
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x[:, -CONTEXT_LENGTH:]
        logits = model(x_cond)[:, -1, :]
        logits = logits / max(temperature, 1e-5)
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)
    return tokenizer.decode(x[0].tolist())


print("=" * 112)
print("Step 44: Step 43B checkpoint inference")
print("=" * 112)

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
if not TOKENIZER.exists():
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:              {device}")
if device.type == "cuda":
    print(f"GPU:                 {torch.cuda.get_device_name(0)}")

checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model = TinyGPT().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = Tokenizer.from_file(str(TOKENIZER))

print(f"checkpoint step:     {checkpoint.get('step')}")
print(f"validation loss:     {checkpoint.get('val_loss')}")
print(f"tokens seen:         {checkpoint.get('tokens_seen'):,}")
print(f"tokenizer vocab:     {tokenizer.get_vocab_size():,}")
print("\nGeneration tests")
print("-" * 112)

prompts = [
    ("English", "The future of artificial intelligence is"),
    ("English", "A good small language model should"),
    ("Chinese", "人工智能的发展将"),
    ("Chinese", "一个好的小型语言模型应该"),
    ("Mixed", "Partdro is a company that"),
    ("Mixed", "一个小型模型可以"),
]

for label, prompt in prompts:
    print(f"\n[{label}] Prompt: {prompt}")
    for temperature in (0.7, 0.9):
        text = generate(model, tokenizer, prompt, device, max_new_tokens=100, temperature=temperature, top_k=40)
        print(f"T={temperature:.1f}: {text}")

print("\nStep 44 complete.")
