#!/usr/bin/env python3
# Nemotron + Dataset Probability Stasis v7 (Entropy Weighted + Visual)
# Core: Entropy-weighted FAISS field + stable semantic pre-generation
# Glow color: pink gradient (#FF8AD8 to #FF1493)

import os, sys, time, numpy as np, pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from colorama import init, Style; init(autoreset=True)
from tqdm import tqdm

# --- Config ---
DATA_FILE = "chat_messages.csv"
MODEL_PATH = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
CTX = 4096; LAYERS = 40; SEED = 42
STABILITY_THRESHOLD = 0.68
PINKS = [(255,138,216),(255,20,147)]  # light to deep pink

_rgb = lambda r,g,b: f"\033[38;2;{r};{g};{b}m"
RESET = Style.RESET_ALL
def grad(val):
    val=max(0,min(1,val)); r=int(PINKS[0][0]+(PINKS[1][0]-PINKS[0][0])*val)
    g=int(PINKS[0][1]+(PINKS[1][1]-PINKS[0][1])*val)
    b=int(PINKS[0][2]+(PINKS[1][2]-PINKS[0][2])*val)
    return _rgb(r,g,b)
def neon(txt): return f"{_rgb(5,217,255)}{txt}{RESET}"
def bar(s,w=22): f=int(w*max(0,min(1,s))); return f"{_rgb(5,217,255)}"+("█"*f+"░"*(w-f))+RESET
def badge(s): return "🧩 STABLE" if s>0.8 else "⚖️ MIXED" if s>0.55 else "🌪️ CHAOTIC"

# --- Load dataset ---
print("📦 Loading dataset...")
if not Path(DATA_FILE).exists():
    print(f"{neon('⚠️')} {DATA_FILE} not found."); sys.exit(1)

df = pd.read_csv(DATA_FILE)
text_col = next((c for c in df.columns if 'body_full' in c.lower() or 'text' in c.lower() or 'message' in c.lower()), None)
user_col = next((c for c in df.columns if 'login' in c.lower() or 'user' in c.lower()), None)
if text_col is None: sys.exit("No text column found.")
texts = df[text_col].astype(str).fillna("")
users = df[user_col].astype(str).fillna("unknown") if user_col else ["unknown"]*len(df)
print(f"🧾 Using text column: {text_col} | user column: {user_col or 'N/A'}")

# --- Embedding model ---
model = SentenceTransformer('all-MiniLM-L6-v2')
print("🔢 Encoding dataset (entropy-weighted FAISS build)...")
embeds = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeds.shape[1])
index.add(embeds)
print(f"✅ FAISS index ready ({len(embeds)} vectors).")

# --- Llama model ---
llm = Llama(model_path=MODEL_PATH, n_ctx=CTX, n_gpu_layers=LAYERS, seed=SEED, verbose=False)

def entropy_weight(similarities):
    p = np.clip(similarities, 1e-8, 1)
    p = p/np.sum(p)
    ent = -np.sum(p*np.log(p))
    return 1/(1+ent)  # higher = purer

def highlight(text, score):
    tokens = text.split()
    return " ".join(f"{grad(min(1,max(0,score)))}{t}{RESET}" for t in tokens)

def compute_stasis(query):
    qv = model.encode([query], normalize_embeddings=True)
    sims, idxs = index.search(qv, 15)
    sims, idxs = sims[0], idxs[0]
    retrieved = [texts[i] for i in idxs]
    stability = 1 - np.std(sims) * 2.2
    entropy_w = entropy_weight(sims)
    final_stability = max(0,min(1,(stability*0.7 + entropy_w*0.3)))
    conf = np.mean(sims)
    return final_stability, conf, retrieved, sims

def generate_response(query, context, stability):
    if stability < STABILITY_THRESHOLD:
        return "The context is too unstable for safe generation. More consistent data is needed."
    ctx_txt = "\n".join(context[:5])
    prompt = f"Context:\n{ctx_txt}\n\nQuestion: {query}\nAnswer clearly and briefly based only on context."
    output = llm(prompt=prompt, max_tokens=180, temperature=0.72, stop=["</s>"])
    return output["choices"][0]["text"].strip()

def main():
    print(f"\n🤖 Nemotron + Dataset Probability Stasis v7 (Entropy + Visual + Safe Gen)\nType 'quit' to exit.\n")
    while True:
        q = input(f"{neon('💬 You: ')}")
        if q.strip().lower() in ['quit','exit']: break
        print(f"\n⚙️ Computing entropy-weighted stasis field...\n")
        st, conf, ctx, sims = compute_stasis(q)
        print(f"🏆 Stability [{st:.3f}] {bar(st)} {badge(st)}")
        print(f"📊 Dataset Confidence [{conf:.3f}]")
        ans = generate_response(q, ctx, st)
        print("\n╭──────────────────────────────────────────────╮")
        print(neon("Nemotron says:"))
        print(ans)
        print("╰──────────────────────────────────────────────╯\n")
        print("🔍 Top dataset evidence:")
        for i,(t,s) in enumerate(zip(ctx[:5], sims[:5])):
            print(f"#{i+1} {highlight(t, s)}")
        print("-"*60)

if __name__ == "__main__":
    main()
