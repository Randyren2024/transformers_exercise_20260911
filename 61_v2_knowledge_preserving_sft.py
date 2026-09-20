import argparse
import math
import random
import time
import re
time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

SEED = 61
VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256
BATCH = 64
STEPS = 1600
LR = 1.0e-6
MIN_LR = 2.0e-7
WARM = 120
WD = 0.01
END = " [END]"

EN_WIKI_DOCS = 2500
ZH_WIKI_DOCS = 2500

DRIVE = Path('/content/drive/MyDrive/transformers_exercise_20260911')
CKPT = DRIVE / 'artifacts' / 'step59' / 'tiny_gpt_v2_best.pt'
TOK_PATH = DRIVE / 'artifacts' / 'step43' / 'step43_bpe_8000.json'
OUT_DIR = DRIVE / 'artifacts' / 'step61'


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(text):
    if not isinstance(text, str):
        return ''
    return re.sub(r'\s+', ' ', text).strip()


def first_sentences(text, max_chars=220):
    text = clean(text)
    if not text:
        return ''
    parts = re.split(r'(?<=[.!?。！？])\s+', text)
    chosen = []
    total = 0
    for part in parts[:3]:
        if not part:
            continue
        if total + len(part) > max_chars:
            break
        chosen.append(part)
        total += len(part) + 1
    result = ' '.join(chosen).strip()
    if len(result) < 20:
        result = text[:max_chars].strip()
    return result[:max_chars].strip()


def wiki_examples(config, limit, lang):
    ds = load_dataset(
        'wikimedia/wikipedia',
        config,
        split='train',
        streaming=True,
    ).shuffle(seed=SEED + (1 if lang == 'en' else 2), buffer_size=5000)

    rows = []
    seen = set()
    for row in ds:
        title = clean(row.get('title'))
        body = clean(row.get('text', ''))
        if not title or len(title) < 2 or len(title) > 90:
            continue
        answer = first_sentences(body)
        if len(answer) < 20:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)

        if lang == 'en':
            prompts = [
                f'Tell me briefly about {title}.',
                f'What is {title}?',
                f'Please briefly describe {title}.',
            ]
        else:
            prompts = [
                f'请简要介绍{title}。',
                f'{title}是什么？',
                f'请简单说明{title}。',
            ]

        prompt = prompts[len(rows) % len(prompts)]
        rows.append((lang, prompt, answer))
        if len(rows) >= limit:
            break

    return rows


def curated_rows():
    rows = []

    exact = [
        ('en', 'Translate \'robot\' into Chinese.', '机器人。'),
        ('en', 'Translate \'data\' into Chinese.', '数据。'),
        ('en', 'Translate \'software\' into Chinese.', '软件。'),
        ('en', 'Translate \'hardware\' into Chinese.', '硬件。'),
        ('en', 'Translate \'computer\' into Chinese.', '电脑。'),
        ('en', 'Translate \'language model\' into Chinese.', '语言模型。'),
        ('en', 'Translate \'machine learning\' into Chinese.', '机器学习。'),
        ('en', 'Translate \'attention\' into Chinese.', '注意力。'),
        ('zh', '把“robot”翻译成中文。', '机器人。'),
        ('zh', '把“data”翻译成中文。', '数据。'),
        ('zh', '把“software”翻译成中文。', '软件。'),
        ('zh', '把“hardware”翻译成中文。', '硬件。'),
        ('zh', '把“computer”翻译成中文。', '电脑。'),
        ('zh', '把“language model”翻译成中文。', '语言模型。'),
        ('zh', '把“machine learning”翻译成中文。', '机器学习。'),
        ('zh', '把“attention”翻译成中文。', '注意力。'),
        ('en', 'What can you do?', 'I can answer simple questions and follow short instructions.'),
        ('zh', '你能做什么？', '我可以回答简单问题并执行简短指令。'),
        ('en', 'Who are you?', 'I am a small bilingual language model.'),
        ('zh', '你是谁？', '我是一个小型双语语言模型。'),
        ('en', 'Are you a human?', 'No. I am a language model.'),
        ('zh', '你是人类吗？', '不是。我是一个语言模型。'),
        ('en', 'What is artificial intelligence?', 'Artificial intelligence is technology that lets computers perform tasks that normally require human intelligence.'),
        ('zh', '什么是人工智能？', '人工智能是让计算机执行通常需要人类智能任务的一种技术。'),
        ('en', 'What is machine learning?', 'Machine learning lets computers learn patterns from data instead of using a separate rule for every case.'),
        ('zh', '什么是机器学习？', '机器学习让计算机从数据中学习规律，而不是为每种情况都手写规则。'),
        ('en', 'What is a language model?', 'A language model predicts likely tokens from context and can generate text.'),
        ('zh', '什么是语言模型？', '语言模型根据上下文预测可能的词元，并生成文本。'),
        ('en', 'What is attention?', 'Attention lets a model focus on relevant parts of the input.'),
        ('zh', '什么是注意力？', '注意力机制让模型重点关注输入中更相关的部分。'),
        ('en', 'What is a transformer?', 'A transformer is a neural network architecture built around attention.'),
        ('zh', '什么是 Transformer？', 'Transformer 是一种以注意力机制为核心的神经网络架构。'),
    ]
    rows.extend(exact)

    facts = [
        ('en', 'How many days are in a week?', 'Seven.'),
        ('en', 'How many months are in a year?', 'Twelve.'),
        ('en', 'What color is grass usually?', 'Green.'),
        ('en', 'What do bees make?', 'Honey.'),
        ('en', 'What is the opposite of hot?', 'Cold.'),
        ('en', 'What is the opposite of big?', 'Small.'),
        ('zh', '一周有几天？', '七天。'),
        ('zh', '一年有几个月？', '十二个月。'),
        ('zh', '草通常是什么颜色？', '绿色。'),
        ('zh', '蜜蜂生产什么？', '蜂蜜。'),
        ('zh', '热的反义词是什么？', '冷。'),
        ('zh', '大的反义词是什么？', '小。'),
    ]
    rows.extend(facts)

    # A small amount of arithmetic teaches answer format, but arithmetic is not
    # allowed to dominate the knowledge-preserving objective.
    for a, b in [(3, 14), (8, 27), (19, 24), (36, 17), (42, 29), (11, 23), (18, 26), (27, 35)]:
        rows.append(('en', f'What is {a} + {b}?', f'{a + b}.'))
        rows.append(('zh', f'{a} 加 {b} 等于多少？', f'{a + b}。'))

    return rows


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

    def forward(self, x, y=None):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def generate(self, x, tok, max_new=48):
        self.eval()
        start = x.size(1)
        for _ in range(max_new):
            logits, _ = self(x[:, -CTX:])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            text = tok.decode(x[0].tolist()[start:]).strip()
            if '[END]' in text:
                return text.split('[END]', 1)[0].strip()
        return tok.decode(x[0].tolist()[start:]).strip()


def make_tensors(tok, rows):
    xs, ys = [], []
    for lang, user, answer in rows:
        prefix = f'User: {user}\nAssistant:'
        full = prefix + ' ' + answer + END
        prefix_ids = tok.encode(prefix).ids
        ids = tok.encode(full).ids[:CTX]
        start = min(len(prefix_ids), len(ids))
        if len(ids) <= start + 1:
            continue
        x = ids[:-1]
        y = ids[1:]
        ignore = max(0, start - 1)
        y[:ignore] = [-100] * ignore
        pad = CTX - len(x)
        if pad > 0:
            x += [0] * pad
            y += [-100] * pad
        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)


def lr_at(step):
    if step <= WARM:
        return LR * step / WARM
    p = min(max((step - WARM) / max(1, STEPS - WARM), 0.0), 1.0)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=STEPS)
    args, _ = parser.parse_known_args()
    seed_all(SEED)

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    if not TOK_PATH.exists():
        raise FileNotFoundError(TOK_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = Tokenizer.from_file(str(TOK_PATH))

    print('=' * 112)
    print('Node 61 — TinyGPT v2 knowledge-preserving SFT')
    print('=' * 112)
    print('device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    print('\nBuilding Wikipedia-grounded instruction examples...')
    en = wiki_examples('20231101.en', EN_WIKI_DOCS, 'en')
    zh = wiki_examples('20231101.zh', ZH_WIKI_DOCS, 'zh')
    curated = curated_rows()
    rows = en + zh + curated

    # Balance languages without discarding the knowledge examples.
    random.shuffle(en)
    random.shuffle(zh)
    n = min(len(en), len(zh))
    balanced = en[:n] + zh[:n] + curated
    random.shuffle(balanced)

    cut = int(len(balanced) * 0.90)
    train_rows = balanced[:cut]
    val_rows = balanced[cut:]
    train_x, train_y = make_tensors(tok, train_rows)
    val_x, val_y = make_tensors(tok, val_rows)

    print(f'Train examples: {len(train_x):,}')
    print(f'Validation examples: {len(val_x):,}')
    print(f'Wikipedia EN: {len(en):,} | ZH: {len(zh):,}')
    print(f'Curated examples: {len(curated):,}')

    model = TinyGPT().to(device)
    state = torch.load(CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = float('inf')
    tick = time.perf_counter()
    n_train = len(train_x)

    for step in range(1, args.steps + 1):
        model.train()
        lr = lr_at(step)
        optimizer.param_groups[0]['lr'] = lr
        idx = torch.randint(0, n_train, (BATCH,))
        x = train_x[idx].to(device)
        y = train_y[idx].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == args.steps:
            speed = step * BATCH * CTX / max(time.perf_counter() - tick, 1e-6)
            print(f'step {step:>4}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s')

        if step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                _, val_loss = model(val_x[:BATCH].to(device), val_y[:BATCH].to(device))
            print(f'validation loss: {val_loss.item():.4f}')

            if val_loss.item() < best:
                best = val_loss.item()
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'step': step,
                        'validation_loss': best,
                        'base_checkpoint': str(CKPT),
                        'tokenizer_path': str(TOK_PATH),
                    },
                    OUT_DIR / 'tiny_gpt_v2_sft_best.pt',
                )
                print('saved new best:', best)

    print('\nPrecision probes')
    probes = [
        '李白是谁？',
        '请介绍一下李白。',
        'What is artificial intelligence?',
        '什么是人工智能？',
        '什么是 Transformer？',
        'What can you do?',
        '你能做什么？',
        'Translate \'software\' into Chinese.',
        '把“data”翻译成中文。',
    ]
    model.eval()
    for user in probes:
        prompt = f'User: {user}\nAssistant:'
        x = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        print(f'\nUser: {user}\nAssistant: {model.generate(x, tok)}')

    print('\nNode 61 complete.')
    print('Best checkpoint:', OUT_DIR / 'tiny_gpt_v2_sft_best.pt')


if __name__ == '__main__':
    main()