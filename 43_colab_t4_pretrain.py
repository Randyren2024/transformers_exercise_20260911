import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CANDIDATE_ROOTS = [SCRIPT_DIR, SCRIPT_DIR / "transformers_exercise_20260911", Path("/content/transformers_exercise_20260911"), Path("/content")]
TRAIN_REL = Path("data/step42_training_corpus/step42_train_ids.pt")
VAL_REL = Path("data/step42_training_corpus/step42_validation_ids.pt")
META_REL = Path("data/step42_training_corpus/step42_metadata.json")

def first_existing_path(relative_path):
    for root in CANDIDATE_ROOTS:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None

TRAIN_IDS = first_existing_path(TRAIN_REL)
VAL_IDS = first_existing_path(VAL_REL)
META_PATH = first_existing_path(META_REL)
ARTIFACT_DIR = SCRIPT_DIR / "artifacts" / "step43"
CHECKPOINT_PATH = ARTIFACT_DIR / "tiny_gpt_step43.pt"
CONFIG_PATH = ARTIFACT_DIR / "tiny_gpt_step43_config.json"

VOCAB_SIZE = 8_000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256
DROPOUT = 0.0
BATCH_SIZE_GPU = 64
BATCH_SIZE_CPU = 8
DEFAULT_STEPS = 5_000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
EVAL_INTERVAL = 250
EVAL_BATCHES = 20
LOG_INTERVAL = 25
SAVE_INTERVAL = 500

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length))

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(~mask, float("-inf"))
            y = torch.softmax(scores, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, context_length, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Linear(d_ff, d_model, bias=False))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock(D_MODEL, N_HEADS, D_FF, CONTEXT_LENGTH, DROPOUT) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, targets=None):
        _, seq_len = input_ids.shape
        if seq_len > CONTEXT_LENGTH:
            raise ValueError("Sequence length exceeds context length")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_batch(tokens, batch_size, device):
    max_start = len(tokens) - CONTEXT_LENGTH - 1
    if max_start < 0: raise RuntimeError("Token corpus is too small for the selected context length.")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[int(start):int(start)+CONTEXT_LENGTH] for start in starts])
    y = torch.stack([tokens[int(start)+1:int(start)+CONTEXT_LENGTH+1] for start in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss(model, tokens, batch_size, device, batches, amp_enabled):
    model.eval(); losses=[]
    for _ in range(batches):
        x,y=get_batch(tokens,batch_size,device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            _,loss=model(x,y)
        losses.append(loss.item())
    model.train(); return sum(losses)/len(losses)

def current_lr(step,total_steps):
    if step<=WARMUP_STEPS: return LEARNING_RATE*step/max(WARMUP_STEPS,1)
    progress=min(max((step-WARMUP_STEPS)/max(total_steps-WARMUP_STEPS,1),0.0),1.0)
    cosine=0.5*(1.0+math.cos(math.pi*progress))
    return MIN_LEARNING_RATE+(LEARNING_RATE-MIN_LEARNING_RATE)*cosine

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--steps",type=int,default=DEFAULT_STEPS); parser.add_argument("--batch-size",type=int,default=None); parser.add_argument("--seed",type=int,default=SEED); args=parser.parse_args()
    set_seed(args.seed); ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
    if TRAIN_IDS is None or VAL_IDS is None:
        print("Step 42 token files were not found in the current Colab environment.")
        for root in CANDIDATE_ROOTS: print(f"  - {root / TRAIN_REL}")
        sys.exit(1)
    train_tokens=torch.load(TRAIN_IDS,map_location="cpu"); val_tokens=torch.load(VAL_IDS,map_location="cpu")
    train_tokens=train_tokens.long(); val_tokens=val_tokens.long()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); cuda_enabled=device.type=="cuda"
    batch_size=args.batch_size or (BATCH_SIZE_GPU if cuda_enabled else BATCH_SIZE_CPU)
    scaler=torch.cuda.amp.GradScaler(enabled=cuda_enabled)
    model=TinyGPT().to(device); params=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,betas=(0.9,0.95))
    print("="*112); print("Step 43: Colab T4 formal TinyGPT pretraining"); print("="*112)
    print("\nPart 1: Corpus\n"+"-"*112); print(f"Train tokens:        {len(train_tokens):,}"); print(f"Validation tokens:   {len(val_tokens):,}")
    print("\nPart 2: Model / hardware\n"+"-"*112); print(f"Parameters:          {params:,}"); print(f"vocab:               {VOCAB_SIZE:,}"); print(f"d_model / heads:     {D_MODEL} / {N_HEADS}"); print(f"layers / FFN:        {N_LAYERS} / {D_FF}"); print(f"context length:      {CONTEXT_LENGTH}"); print(f"device:              {device}")
    if cuda_enabled: print(f"GPU:                 {torch.cuda.get_device_name(0)}\nmixed precision:     True (float16 autocast)")
    else: print("mixed precision:     False")
    print(f"training steps:      {args.steps:,}"); print(f"batch size:          {batch_size}"); print(f"effective tokens/step: {batch_size*CONTEXT_LENGTH:,}"); print("checkpoint policy:   clean baseline; no Step 40/41 checkpoint loaded")
    initial_train=estimate_loss(model,train_tokens,batch_size,device,5,cuda_enabled); initial_val=estimate_loss(model,val_tokens,batch_size,device,5,cuda_enabled)
    print(f"Initial train loss:  {initial_train:.4f}"); print(f"Initial val loss:    {initial_val:.4f}")
    print("\nPart 3: Formal pretraining\n"+"-"*112); model.train(); tokens_seen=0; latest_train=initial_train; latest_val=initial_val; timer=time.perf_counter(); last_log_step=0
    for step in range(1,args.steps+1):
        lr=current_lr(step,args.steps)
        for group in optimizer.param_groups: group["lr"]=lr
        x,y=get_batch(train_tokens,batch_size,device); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=cuda_enabled): _,loss=model(x,y)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); scaler.step(optimizer); scaler.update()
        latest_train=loss.item(); tokens_seen+=batch_size*CONTEXT_LENGTH
        if step==1 or step%LOG_INTERVAL==0 or step==args.steps:
            elapsed=max(time.perf_counter()-timer,1e-6); speed=(step-last_log_step)*batch_size*CONTEXT_LENGTH/elapsed; timer=time.perf_counter(); last_log_step=step
            print(f"step {step:>5}/{args.steps} | loss {latest_train:.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")
        if step%EVAL_INTERVAL==0 or step==args.steps:
            latest_val=estimate_loss(model,val_tokens,batch_size,device,EVAL_BATCHES,cuda_enabled); print(f"           validation loss: {latest_val:.4f}")
        if step%SAVE_INTERVAL==0 or step==args.steps:
            config={"step":step,"vocab_size":VOCAB_SIZE,"d_model":D_MODEL,"n_heads":N_HEADS,"n_layers":N_LAYERS,"d_ff":D_FF,"context_length":CONTEXT_LENGTH,"batch_size":batch_size,"learning_rate":LEARNING_RATE,"min_learning_rate":MIN_LEARNING_RATE,"warmup_steps":WARMUP_STEPS,"weight_decay":WEIGHT_DECAY,"weight_tying":True,"parameter_count":params,"tokens_seen":tokens_seen,"device":str(device),"amp_enabled":cuda_enabled,"seed":args.seed,"train_ids_path":str(TRAIN_IDS),"val_ids_path":str(VAL_IDS)}
            torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scaler_state_dict":scaler.state_dict(),"step":step,"train_loss":latest_train,"val_loss":latest_val,"tokens_seen":tokens_seen,"config":config},CHECKPOINT_PATH); print(f"           checkpoint saved: {CHECKPOINT_PATH}")
    final_config={"step":args.steps,"parameter_count":params,"train_tokens":len(train_tokens),"validation_tokens":len(val_tokens),"tokens_seen":tokens_seen,"batch_size":batch_size,"device":str(device),"amp_enabled":cuda_enabled,"initial_train_loss":initial_train,"initial_val_loss":initial_val,"final_train_loss":latest_train,"final_val_loss":latest_val,"checkpoint":str(CHECKPOINT_PATH)}
    CONFIG_PATH.write_text(json.dumps(final_config,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nPart 4: Results\n"+"-"*112); print(f"Initial train loss:  {initial_train:.4f}"); print(f"Final train loss:    {latest_train:.4f}"); print(f"Initial val loss:    {initial_val:.4f}"); print(f"Final val loss:      {latest_val:.4f}"); print(f"Tokens seen:         {tokens_seen:,}"); print(f"Checkpoint saved:    {CHECKPOINT_PATH}"); print(f"Config saved:        {CONFIG_PATH}"); print("\nStep 43 complete.")

if __name__=="__main__": main()
