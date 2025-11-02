#!/usr/bin/env python3
# Nemotron + Probability Stasis v3 (Dataset-Aware, FAISS, #05D9FF Glow)
# 🧠 Dynamically detects best text column and actually grounds answers.

import os, sys, time, numpy as np, pandas as pd, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from colorama import Style, init; init(autoreset=True)

DATA_FILE = "chat_messages.csv"
MODEL_PATH = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
CTX, LAYERS, SEED = 4096, 40, 42
NEON = (5,217,255); _rgb=lambda r,g,b:f"\033[38;2;{r};{g};{b}m"; RESET=Style.RESET_ALL
def neon(t):return f"{_rgb(*NEON)}{t}{RESET}"
def bar(s,w=22):f=int(w*max(0,min(1,s)));return f"{_rgb(*NEON)}"+("█"*f+"░"*(w-f))+RESET
def badge(s):return"🧩 STABLE"if s>0.8 else"⚖️ MIXED"if s>0.55 else"🌪️ CHAOTIC"

# === Load Dataset ===
if not Path(DATA_FILE).exists():
    print(f"{neon('⚠️')} {DATA_FILE} not found.");sys.exit(1)
print("📦 Loading dataset...")
df=pd.read_csv(DATA_FILE)

# Auto-detect likely text column
text_candidates=[c for c in df.columns if any(k in c.lower() for k in ['message','text','body','chat','content','msg'])]
if not text_candidates:
    print("❌ No text-like column found.");sys.exit(1)
TEXT_COL=text_candidates[0]
print(f"🧾 Using text column: {neon(TEXT_COL)} ({len(df)} rows)")

texts=df[TEXT_COL].astype(str).fillna("")

# === Encode & Build FAISS ===
print("🔢 Encoding dataset (FAISS index build)...")
embedder=SentenceTransformer("all-MiniLM-L6-v2")
emb=embedder.encode(texts,normalize_embeddings=True,show_progress_bar=True)
index=faiss.IndexFlatIP(emb.shape[1]);index.add(np.array(emb).astype("float32"))
print(f"✅ FAISS index built ({len(emb)} vectors).")

# === Load Model ===
llm=Llama(model_path=MODEL_PATH,n_ctx=CTX,n_gpu_layers=LAYERS,seed=SEED,verbose=False)
SYS="You are Nemotron, a factual reasoning model grounded in dataset context. Use only context for answers."

def query_dataset(q):
    q_emb=embedder.encode([q],normalize_embeddings=True)
    D,I=index.search(np.array(q_emb).astype("float32"),k=5)
    contexts=[texts[i] for i in I[0] if isinstance(texts[i],str)]
    return contexts, float(D.mean())

print(f"🤖 Nemotron + Dataset Probability Stasis v3 (FAISS, {neon('#05D9FF')})")
print("Type 'quit' to exit.\n")

while True:
    try:
        q=input(f"{neon('💬 You: ')}").strip()
        if not q or q.lower() in ["quit","exit"]:break
        print(f"{neon('⚙️ Computing dataset stasis (FAISS)...')}")
        ctxs,score=query_dataset(q)
        stability=(score+1)/2
        joined="\n".join(ctxs)
        prompt=f"{SYS}\n\nContext:\n{joined}\n\nQuestion: {q}\nAnswer:"
        out=llm(prompt,max_tokens=200,temperature=0.7,stop=["</think>"])
        ans=out["choices"][0]["text"].strip()
        print(f"\n🏆 Stability [{neon(f'{stability:.3f}')}] {bar(stability)} {badge(stability)}\n")
        print("╭"+"─"*46+"╮")
        print(neon("Nemotron says:"))
        print(ans)
        print("╰"+"─"*46+"╯")
        print("------------------------------------------------------------")
    except KeyboardInterrupt:break
