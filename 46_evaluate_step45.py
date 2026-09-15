import json
import math
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Step 46: evaluate the Step 45 checkpoint with controlled generation.
# This is an inference/evaluation step, not more pretraining.

VOCAB_SIZE = 8000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256

DRIVE_ROOT = Path('/content/drive/MyDrive/transformers_exercise_20260911')
CHECKPOINT = DRIVE_ROOT / 'artifacts' / 'step45' / 'tiny_gpt_step45.pt'
TOKENIZER = DRIVE_ROOT / 'artifacts' / 'step43' / 'step43_bpe_8000.json'

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
    def generate(self, ids, new_tokens=80, temperature=0.7, top_k=40):
        self.eval()
        for _ in range(new_tokens):
            x = ids[:, -CONTEXT_LENGTH:]
            logits = self(x)[:, -1, :] / max(temperature, 1e-5)
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not TOKENIZER.exists():
        raise FileNotFoundError(TOKENIZER)

    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model = TinyGPT()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    tok = Tokenizer.from_file(str(TOKENIZER))

    print('=' * 112)
    print('Step 46: Step 45 checkpoint evaluation')
    print('=' * 112)
    print(f'device:              {device}')
    if device.type == 'cuda':
        print(f'GPU:                 {torch.cuda.get_device_name(0)}')
    print(f'checkpoint step:     {ckpt.get("step")}')
    print(f'tokens seen:         {ckpt.get("tokens_seen"):,.0f}')
    print(f'checkpoint val loss: {ckpt.get("val_loss")}')

    prompts = [
        ('English', 'The future of artificial intelligence is'),
        ('English', 'A good small language model should'),
        ('Chinese', '人工智能的发展将'),
        ('Chinese', '一个好的小型语言模型应该'),
        ('Mixed', '一个小型模型可以'),
        ('Product', 'Partdro is a company that'),
        ('Knowledge', 'A transformer model learns language by'),
        ('Instruction', 'User: 请解释什么是人工智能。\\nAssistant:'),
    ]

    print('\\nGeneration tests')
    print('-' * 112)
    for name, prompt in prompts:
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        for temp in (0.6, 0.75):
            torch.manual_seed(100 + len(prompt) + int(temp * 100))
            out = model.generate(ids.clone(), new_tokens=90, temperature=temp, top_k=40)
            text = tok.decode(out[0].tolist())
            print(f'\\n[{name}] T={temp}: {text}')

    print('\\nStep 46 complete.')

if __name__ == '__main__':
    main()
