#!/usr/bin/env python3
# Nemotron Probability Stasis v6 (Safe Sequential GPU version)
# Fix: prevent CUDA allocator assertion by serializing path runs.

import time, json, csv, sys
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import numpy as np
from colorama import Fore, Style, init
init(autoreset=True)

MODEL_PATH="Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
LOG_FILE=Path("stasis_log.csv")
TEMPS=[0.6,0.7,0.8,0.85,0.9]
MAX_TOKENS=256
SYSTEM_PROMPT="""You are Nemotron, a factual reasoning model.
When asked a question:
1. Think briefly before you answer.
2. Provide your reasoning clearly and concisely.
3. Give a final, verified answer."""

print("🤖  Nemotron + Probability Stasis v6_safe (Sequential GPU)")
llm=Llama(model_path=MODEL_PATH,n_ctx=4096,n_gpu_layers=40,verbose=False)
embedder=SentenceTransformer('all-MiniLM-L6-v2')
_cache={}

def embed(t):
    if t in _cache: return _cache[t]
    e=embedder.encode([t],normalize_embeddings=True)[0]; _cache[t]=e; return e

def color_stab(s):
    if s<0.5:c="\033[96m"
    elif s<0.8:c="\033[94m"
    else:c="\033[95m"
    return f"{c}{s:.3f}{Style.RESET_ALL}"

def badge(s):return "🧩 STABLE" if s>=0.8 else ("⚖️ MIXED" if s>=0.5 else "🌪️ CHAOTIC")

def run_paths(prompt):
    outs=[]
    for t in TEMPS:
        out=llm(f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:",temperature=t,max_tokens=MAX_TOKENS)
        outs.append({"t":t,"txt":out["choices"][0]["text"].strip()})
    return outs

def choose(outs):
    embs=np.stack([embed(o["txt"]) for o in outs])
    sims=util.cos_sim(embs,embs).cpu().numpy()
    mask=~np.eye(len(outs),dtype=bool)
    stab=float(sims[mask].mean()-sims[mask].std())
    mean_sims=sims.sum(1)-1
    win=int(np.argmax(mean_sims))
    return outs[win]["txt"],outs[win]["t"],stab

def loop():
    while True:
        q=input("💬 You: ").strip()
        if q in {"quit","exit"}:break
        t0=time.time()
        outs=run_paths(q)
        txt,temp,stab=choose(outs)
        dur=time.time()-t0
        print(f"\n🏆 Stability [{color_stab(stab)}] {'█'*int(stab*20):<20} {badge(stab)}")
        print("╭──────────────────────────────────────────────╮")
        print("Nemotron says:\n"+txt)
        print("╰──────────────────────────────────────────────╯")
        print(f"⏱️ {dur:.1f}s | Temp {temp}")
        with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([time.strftime("%F %T"),q,f"{stab:.4f}",temp,f"{dur:.2f}"])
        print("------------------------------------------------------------")

if __name__=="__main__": loop()
