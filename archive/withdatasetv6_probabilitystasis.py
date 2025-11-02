#!/usr/bin/env python3
# Nemotron + Probability Stasis v6 (Dataset-Aware + Pink Gradient Highlights)
# Adds: top snippet display + dataset confidence metric + pink gradient word highlights

import os, sys, time, json, math, re, hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import faiss

# === CONFIG ===
DATA_FILE   = "chat_messages.csv"
MODEL_PATH  = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
EMB_MODEL   = "all-MiniLM-L6-v2"
CTX, LAYERS, SEED = 4096, 40, 42
TOP_K = 15
NEON = (5, 217, 255)
PINKS = [(255, 182, 249), (255, 120, 210), (255, 46, 166)]
RESET = "\033[0m"
CACHE_DIR = Path(".stasis_cache")

def _rgb(r,g,b): return f"\033[38;2;{r};{g};{b}m"
def neon(txt): return f"{_rgb(*NEON)}{txt}{RESET}"
def bar(s,w=22): s=max(0,min(1,float(s))); f=int(w*s); return _rgb(*NEON)+("█"*f+"░"*(w-f))+RESET
def badge(s): return "🧩 STABLE" if s>=0.8 else "⚖️ MIXED" if s>=0.55 else "🌪️ CHAOTIC"

def gradient_word(word, weight):
    # map [0,1] → gradient from light→dark pink
    weight = max(0.0, min(1.0, weight))
    c = tuple(int(PINKS[0][i] + (PINKS[-1][i]-PINKS[0][i])*weight) for i in range(3))
    return f"{_rgb(*c)}{word}{RESET}"

def highlight_relevant_words(text, query_emb, embedder):
    words = re.findall(r"\w+|\s+|[^\w\s]", text)
    tokens = [w for w in words if w.strip()]
    if not tokens: return text
    vecs = embedder.encode(tokens, convert_to_numpy=True, normalize_embeddings=True)
    sims = np.dot(vecs, query_emb.squeeze())
    sims = (sims - sims.min()) / (sims.max()-sims.min()+1e-9)
    highlighted = []
    idx = 0
    for w in words:
        if w.strip():
            highlighted.append(gradient_word(w, sims[idx]**2))
            idx += 1
        else:
            highlighted.append(w)
    return "".join(highlighted)

def build_or_load_index(df, text_col, user_col):
    sig = hashlib.md5(f"{DATA_FILE}-{text_col}-{user_col}-{EMB_MODEL}".encode()).hexdigest()
    emb_path = CACHE_DIR/f"emb_{sig}.npy"
    idx_path = CACHE_DIR/f"faiss_{sig}.index"
    CACHE_DIR.mkdir(exist_ok=True)
    embedder = SentenceTransformer(EMB_MODEL)

    corpus = [f"user={u} :: message={t}" for u,t in zip(df[user_col].astype(str), df[text_col].astype(str))]
    if emb_path.exists() and idx_path.exists():
        embs = np.load(emb_path)
        index = faiss.read_index(str(idx_path))
        print("✅ FAISS index cached.")
        return corpus, embs, index, embedder

    print("🔢 Encoding dataset...")
    vecs = embedder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    embs = np.asarray(vecs, dtype="float32")
    np.save(emb_path, embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(idx_path))
    print(f"✅ Built FAISS index ({len(corpus)} vectors).")
    return corpus, embs, index, embedder

def stability_score(sim):
    mu, sd = sim.mean(), sim.std()
    return mu - 0.5*sd

def call_llm(llm, sys_prompt, prompt, temp):
    out = llm(prompt=f"{sys_prompt}\n\n{prompt}\nAssistant:", temperature=temp, max_tokens=256, stop=["User:","Assistant:"])
    return out["choices"][0]["text"].strip()

def clamp01(x): return max(0.0, min(1.0, float(x)))

def main():
    if not Path(DATA_FILE).exists():
        print(f"{neon('⚠️')} {DATA_FILE} not found.")
        return
    df = pd.read_csv(DATA_FILE)
    text_col = "body_full" if "body_full" in df.columns else "body"
    user_col = "login" if "login" in df.columns else "user_id"
    corpus, embs, index, embedder = build_or_load_index(df, text_col, user_col)
    llm = Llama(model_path=MODEL_PATH, n_ctx=CTX, n_gpu_layers=LAYERS, seed=SEED, verbose=False)

    print(neon("\n🤖 Nemotron + Dataset Probability Stasis v6 (Pink Gradient Highlights)"))
    print("Type 'quit' to exit.\n")

    SYS = ("You are Nemotron, a factual model. Use the dataset context provided to answer accurately.\n"
           "If unknown, say 'unknown from dataset.' Keep answers under 3 sentences.")

    while True:
        q = input(neon("💬 You: ")).strip()
        if not q or q.lower() in {"quit","exit"}: break

        q_emb = embedder.encode([f"question={q}"], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D,I = index.search(q_emb, TOP_K)
        sims, idxs = D[0], I[0]
        mean_conf = sims.mean()
        top_contexts = [corpus[i] for i in idxs if i>=0]

        # build context block
        context_block = "\n".join(top_contexts[:TOP_K])
        path_outputs = []
        for t in [0.62,0.72,0.82,0.9]:
            prompt = f"Context:\n{context_block}\n\nQuestion: {q}\nAnswer:"
            path_outputs.append(call_llm(llm, SYS, prompt, t))

        stab = stability_score(np.array([sims]))
        s_norm = clamp01((stab+1)/2)
        best = path_outputs[int(np.random.randint(0,len(path_outputs)))]

        print(f"\n🏆 Stability [{neon(f'{stab:.3f}')}]", bar(s_norm), badge(s_norm))
        print(f"📊 Dataset Confidence [{neon(f'{mean_conf:.3f}')}]\n")
        print("╭" + "─"*46 + "╮")
        print("Nemotron says:")
        print(best)
        print("╰" + "─"*46 + "╯")

        # show top 3 context snippets with pink highlights
        print(neon("\n🔍 Top dataset evidence (highlighted):"))
        for j, line in enumerate(top_contexts[:3],1):
            msg = line.split(":: message=",1)[-1]
            highlighted = highlight_relevant_words(msg, q_emb, embedder)
            print(f"{_rgb(*PINKS[-1])}#{j}{RESET} {highlighted}")
        print("-"*60)

if __name__=="__main__":
    main()
