import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        hd = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.hd = hd

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, H, self.hd).transpose(1, 2)
        k = k.view(b, t, H, self.hd).transpose(1, 2)
        v = v.view(b, t, H, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, D))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FF, bias=False)
        self.fc2 = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

class TinyGPTService:
    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.model = TinyGPT().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def generate(self, message, max_new_tokens=64):
        prompt = f"User: {message.strip()}\nAssistant:"
        ids = self.tokenizer.encode(prompt).ids[-CTX:]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        start = x.size(1)
        for _ in range(max_new_tokens):
            logits = self.model(x[:, -CTX:])[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_id], dim=1)
            generated = self.tokenizer.decode(x[0].tolist()[start:]).strip()
            if "[END]" in generated:
                return generated.split("[END]", 1)[0].strip()
        return self.tokenizer.decode(x[0].tolist()[start:]).strip()