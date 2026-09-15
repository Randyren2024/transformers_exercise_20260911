import argparse
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

SEED=48; V=8000; D=192; H=6; L=6; FF=768; T=256; B=64
STEPS=2200; LR=2e-5; MIN_LR=2e-6; WARM=150; WD=0.01


def seed_all(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def add(rows,lang,u,a):
    u=re.sub(r"\s+"," ",u).strip(); a=re.sub(r"\s+"," ",a).strip()
    if 3<=len(u)<=240 and 4<=len(a)<=420: rows.append((lang,u,a))


def build():
    rows=[]
    facts=[
      ("What is artificial intelligence?","Artificial intelligence is the field of building computer systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing images, and reasoning.","什么是人工智能？","人工智能是让计算机执行通常需要人类智能的任务，例如理解语言、识别图像和进行推理。"),
      ("What is machine learning?","Machine learning lets computers learn patterns from data instead of using a separate hand-written rule for every case.","什么是机器学习？","机器学习让计算机从数据中学习规律，而不是为每一种情况都手写规则。"),
      ("What is a transformer model?","A transformer is a neural network architecture built around attention and widely used for modern language models.","什么是 Transformer 模型？","Transformer 是一种以注意力机制为核心的神经网络架构，广泛用于现代语言模型。"),
      ("What is an algorithm?","An algorithm is a clear sequence of steps used to solve a problem or complete a task.","什么是算法？","算法是一组清晰、有顺序的步骤，用来解决问题或完成任务。"),
      ("What is a token?","A token is a small unit of text processed by a language model. It can be a word, part of a word, or punctuation.","什么是 token？","Token 是语言模型处理的文本小单元，可以是一个词、词的一部分或标点。"),
      ("What is training?","Training adjusts a model's parameters so that its predictions better match examples in the training data.","什么是模型训练？","模型训练是调整模型参数，使预测逐渐更符合训练数据中的例子。"),
      ("What is inference?","Inference means using a trained model to produce an output for new input.","什么是推理？","推理是使用已经训练好的模型，根据新的输入产生输出。"),
      ("What is a neural network?","A neural network is a model made of layers of numerical transformations that can learn patterns from data.","什么是神经网络？","神经网络是一种由多层数值变换组成的模型，可以从数据中学习规律。"),
      ("What is overfitting?","Overfitting happens when a model learns the training examples too closely and works worse on new data.","什么是过拟合？","过拟合是模型过度记住训练样本，导致它在新数据上的表现变差。"),
      ("What is a GPU?","A GPU is a processor designed for many parallel numerical operations and is widely used for machine learning.","什么是 GPU？","GPU 是擅长并行数值运算的处理器，因此广泛用于机器学习。"),
      ("What is an API?","An API is an interface that lets different software systems communicate and exchange data or functions.","什么是 API？","API 是一种接口，让不同的软件系统能够通信并交换数据或调用功能。"),
      ("What is a database?","A database is an organized collection of data that can be stored, searched, and updated efficiently.","什么是数据库？","数据库是有组织的数据集合，可以高效地存储、查询和更新信息。"),
      ("What is Python?","Python is a general-purpose programming language known for readable syntax and a large library ecosystem.","Python 是什么？","Python 是一种通用编程语言，以语法易读和丰富的库生态而闻名。"),
      ("What is cloud computing?","Cloud computing means using computing resources such as servers and storage over a network.","什么是云计算？","云计算是通过网络使用服务器、存储等计算资源。"),
      ("What is the internet?","The internet is a global network of interconnected computer networks that exchange information.","什么是互联网？","互联网是由许多相互连接的计算机网络组成的全球网络，用来交换信息。"),
    ]
    for eqa in facts: add(rows,'en',eqa[0],eqa[1]); add(rows,'zh',eqa[2],eqa[3])
    for a in range(2,21):
      for b in range(2,11):
        add(rows,'en',f"What is {a} + {b}?",f"{a+b}."); add(rows,'zh',f"{a} 加 {b} 等于多少？",f"{a+b}。")
        add(rows,'en',f"What is {a} - {b}?",f"{a-b}."); add(rows,'zh',f"{a} 减 {b} 等于多少？",f"{a-b}。")
    for a in range(2,13):
      for b in range(2,9):
        add(rows,'en',f"What is {a} times {b}?",f"{a*b}."); add(rows,'zh',f"{a} 乘以 {b} 等于多少？",f"{a*b}。")
    tasks=[
      ('Say hello in a friendly way.','Hello! It is nice to meet you.','用友好的方式说你好。','你好！很高兴认识你。'),
      ('Say thank you politely.','Thank you very much.','礼貌地说谢谢。','非常感谢你。'),
      ('Translate "good morning" into Chinese.','早上好。','把“good morning”翻译成英文。','Good morning.'),
      ('Translate "你好" into English.','Hello.','把“thank you”翻译成中文。','谢谢。'),
      ('Give me one simple study tip.','Study one small topic at a time and review it regularly.','给我一个简单的学习建议。','一次学习一个小主题，并定期复习。'),
      ('Give me one simple programming tip.','Test small pieces of code before combining them into a larger program.','给我一个简单的编程建议。','先测试小段代码，再把它们组合成更大的程序。'),
      ('Why is sleep important? Answer in one sentence.','Sleep helps the body and brain recover and function well.','为什么睡眠重要？用一句话回答。','睡眠可以帮助身体和大脑恢复，并保持良好状态。'),
      ('Why is exercise useful? Answer in one sentence.','Regular exercise can improve fitness, mood, and overall health.','为什么运动有用？用一句话回答。','规律运动可以改善体能、情绪和整体健康。'),
      ('Hello, introduce yourself briefly.','Hello! I am a small language model designed to answer simple questions and follow short instructions.','你好，请简单介绍一下你自己。','你好！我是一个小型语言模型，用来回答简单问题并执行简短指令。'),
      ('Can you help me learn?','Yes. Ask one clear question at a time, and we can work through it step by step.','你可以帮助我学习吗？','可以。你可以一次问一个清晰的问题，我们可以一步一步解决。'),
      ('Can you know every fact?','No. A language model can make mistakes and does not know every fact.','你知道所有事实吗？','不知道。语言模型可能犯错，也不可能知道所有事实。'),
      ('What should you do when you are unsure?','Say that you are not sure instead of pretending to know the answer.','不确定答案时应该怎么办？','应该说明自己不确定，而不是假装知道答案。'),
    ]
    for e in tasks: add(rows,'en',e[0],e[1]); add(rows,'zh',e[2],e[3])
    seen=set(); clean=[]
    for r in rows:
      if (r[0],r[1],r[2]) not in seen: seen.add((r[0],r[1],r[2])); clean.append(r)
    en=[r for r in clean if r[0]=='en']; zh=[r for r in clean if r[0]=='zh']; n=min(len(en),len(zh),1200)
    data=en[:n]+zh[:n]; random.shuffle(data); cut=int(len(data)*.9); return data[:cut],data[cut:]


class Attn(nn.Module):
    def __init__(self):
        super().__init__(); hd=D//H; self.qkv=nn.Linear(D,3*D,bias=False); self.out=nn.Linear(D,D,bias=False); self.hd=hd
    def forward(self,x):
        b,t,_=x.shape; q,k,v=self.qkv(x).chunk(3,-1); q=q.view(b,t,H,self.hd).transpose(1,2); k=k.view(b,t,H,self.hd).transpose(1,2); v=v.view(b,t,H,self.hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.out(y.transpose(1,2).contiguous().view(b,t,D))

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=Attn(); self.ln2=nn.LayerNorm(D); self.ffn=nn.Sequential(nn.Linear(D,FF,bias=False),nn.GELU(),nn.Linear(FF,D,bias=False))
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); x=x+self.ffn(self.ln2(x)); return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(V,D); self.pos=nn.Embedding(T,D); self.blocks=nn.ModuleList([Block() for _ in range(L)]); self.ln=nn.LayerNorm(D); self.head=nn.Linear(D,V,bias=False); self.head.weight=self.tok.weight
    def forward(self,x,y=None):
        t=x.size(1); p=torch.arange(t,device=x.device); h=self.tok(x)+self.pos(p)[None,:,:]
        for block in self.blocks: h=block(h)
        logits=self.head(self.ln(h)); loss=None
        if y is not None: loss=F.cross_entropy(logits.reshape(-1,V),y.reshape(-1),ignore_index=-100)
        return logits,loss
    @torch.no_grad()
    def generate(self,x,tok,max_new=70,temperature=.25,top_k=8):
        start=x.size(1); out=''
        for _ in range(max_new):
            logits,_=self(x[:,-T:]); z=logits[:,-1,:]/temperature; vals,_=torch.topk(z,min(top_k,V)); z=z.masked_fill(z<vals[:,-1,None],float('-inf')); nxt=torch.multinomial(torch.softmax(z,-1),1); x=torch.cat([x,nxt],1)
            out=tok.decode(x[0].tolist()[start:])
            if '\nUser:' in out or '\nAssistant:' in out or out.count('你')>18 or out.count('the')>12: break
        return out.strip()


def lr_at(s,total):
    if s<=WARM:return LR*s/WARM
    p=min(1,max(0,(s-WARM)/max(1,total-WARM))); return MIN_LR+(LR-MIN_LR)*.5*(1+math.cos(math.pi*p))


def tensors(tok,rows):
    X=[];Y=[]
    for _,u,a in rows:
        prefix=f'User: {u}\nAssistant:'; full=prefix+' '+a; ids=tok.encode(full).ids; pids=tok.encode(prefix).ids
        ids=ids[:T]; rs=min(len(pids),len(ids));
        if len(ids)<=rs+1: continue
        x=ids[:-1]; y=ids[1:]; y[:max(0,rs-1)]=[-100]*max(0,rs-1); pad=T-len(x); x += [0]*pad; y += [-100]*pad; X.append(torch.tensor(x)); Y.append(torch.tensor(y))
    return torch.stack(X),torch.stack(Y)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=STEPS); args,_=ap.parse_known_args(); seed_all(SEED)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('='*112); print('Step 48: Curated bilingual instruction alignment'); print('='*112); print('device:',dev); print('GPU:',torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU')
    drive=Path('/content/drive/MyDrive/transformers_exercise_20260911'); ck=drive/'artifacts/step45/tiny_gpt_step45.pt'; tp=drive/'artifacts/step43/step43_bpe_8000.json'; tok=Tokenizer.from_file(str(tp))
    tr,va=build(); tx,ty=tensors(tok,tr); vx,vy=tensors(tok,va); print('\nPart 1: Curated SFT dataset'); print('-'*112); print(f'Train examples:      {len(tx):,}'); print(f'Validation examples: {len(vx):,}'); print(f'English / Chinese:   {sum(r[0]=="en" for r in tr):,} / {sum(r[0]=="zh" for r in tr):,}')
    m=TinyGPT().to(dev); m.load_state_dict(torch.load(ck,map_location='cpu')['model_state_dict']); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD,betas=(.9,.95)); scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda'); tick=time.perf_counter(); n=len(tx); m.train()
    for s in range(1,args.steps+1):
        lr=lr_at(s,args.steps); opt.param_groups[0]['lr']=lr; idx=torch.randint(0,n,(B,)); x=tx[idx].to(dev); y=ty[idx].to(dev); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=dev.type=='cuda'): _,loss=m(x,y)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); scaler.step(opt); scaler.update()
        if s==1 or s%25==0 or s==args.steps:
            speed=s*B*T/max(time.perf_counter()-tick,1e-6); print(f'step {s:>5}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s')
        if s%200==0 or s==args.steps:
            m.eval(); mm=min(B,len(vx));
            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=dev.type=='cuda'): _,vl=m(vx[:mm].to(dev),vy[:mm].to(dev))
            print(f'           validation loss: {vl.item():.4f}'); m.train()
    out=drive/'artifacts/step48'; out.mkdir(parents=True,exist_ok=True); outck=out/'tiny_gpt_step48_curated_sft.pt'; outtok=out/'step48_tokenizer.json'; tok.save(str(outtok)); torch.save({'model_state_dict':m.state_dict(),'tokenizer_path':str(outtok),'base_checkpoint':str(ck),'step':args.steps,'train_examples':len(tx),'validation_examples':len(vx)},outck)
    print('\nPart 2: deterministic instruction probes'); prompts=['User: Explain what artificial intelligence is in simple terms.\nAssistant:','User: 请用简单中文解释什么是人工智能。\nAssistant:','User: 你好，请介绍一下你自己。\nAssistant:','User: What is a transformer model?\nAssistant:','User: What is 7 + 8?\nAssistant:','User: 12 加 9 等于多少？\nAssistant:']; m.eval()
    for p in prompts:
        x=torch.tensor([tok.encode(p).ids],device=dev); print('\n'+p+'\n'+m.generate(x,tok))
    print('\nStep 48 complete.'); print('Checkpoint:',outck)

if __name__=='__main__': main()
