#!/usr/bin/env python3
# Nemotron + Probability Stasis (Dataset v1)
# Uses chat_messages.csv as a probabilistic substrate
# 🧠 Dataset-grounded reasoning + #05D9FF neon output

import os, sys, time, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from colorama import Style, init; init(autoreset=True)

# --- Config ---
DATA_FILE = "chat_messages.csv"
MODEL_PATH = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
CTX = 4096; LAYERS = 40; SEED = 42
NEON = (5,217,255)   # #05D9FF pure neon
_rgb = lambda r,g,b: f"\033[38;2;{r};{g};{b}m"
RESET = Style.RESET_ALL

# --- Neon UI helpers ---
def neon(txt): return f"{_rgb(*NEON)}{txt}{RESET}"
def bar(s,w=22): f=int(w*max(0,min(1,s))); return f"{_rgb(*NEON)}"+("█"*f+"░"*(w-f))+RESET
def badge(s): return "🧩 STABLE" if s>0.8 else "⚖️ MIXED" if s>0.55 else "🌪️ CHAOTIC"

# --- Load / embed dataset ---
if not Path(DATA_FILE).exists():
    print(f"{neon('⚠️')} {DATA_FILE} not found. Please place it in this directory.")
    sys.exit(1)

print(neon("📦 Loading dataset..."))
df = pd.read_csv(DATA_FILE)
text_col = [c for c in df.columns if 'message' in c.lower() or 'text' in c.lower()][0]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(neon("🔢 Encoding dataset (this may take a moment)..."))
df["embedding"] = df[text_col].apply(lambda x: embedder.encode(str(x), normalize_embeddings=True))

# --- Build probability stasis core ---
def probability_stasis(query, top_k=5):
    qv = embedder.encode(query, normalize_embeddings=True)
    embs = np.vstack(df["embedding"].values)
    sims = util.cos_sim(qv, embs).cpu().numpy()[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_msgs = df.iloc[top_idx][text_col].tolist()
    weights = sims[top_idx] / sims[top_idx].sum()
    stability = float(weights.mean() - weights.std())
    return list(zip(top_msgs, weights)), stability

# --- LLM Setup ---
LLM = Llama(model_path=MODEL_PATH, n_ctx=CTX, n_gpu_layers=LAYERS, seed=SEED, verbose=False)
SYS = "You are Nemotron, a factual reasoning model. Answer based only on the weighted dataset messages."

def ask(query):
    stasis, stability = probability_stasis(query)
    context = "\n".join([f"{w:.2f} :: {m}" for m,w in stasis])
    prompt = f"""{SYS}

Dataset-weighted context:
{context}

User question: {query}
Assistant:"""
    out = LLM(prompt, temperature=0.7, max_tokens=256, stop=["User:"])
    ans = out["choices"][0]["text"].strip()
    return ans, stability

# --- Main Loop ---
print(neon("🤖 Nemotron + Dataset Probability Stasis v1 (#05D9FF Glow)"))
print("Type 'quit' to exit.\n")

while True:
    try:
        q = input("💬 You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not q or q.lower() in {"quit","exit"}:
        break
    print(neon("⚙️ Computing dataset stasis..."))
    ans, st = ask(q)
    print(f"\n🏆 Stability [{neon(f'{st:.3f}')}] {bar(st)} {badge(st)}\n")
    print("╭"+"─"*46+"╮")
    print("Nemotron says:")
    print(ans)
    print("╰"+"─"*46+"╯")
    print("-"*60)
