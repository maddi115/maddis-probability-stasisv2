#!/usr/bin/env python3
# Hybrid Semantic + Keyword Search

import os, sys, time, json, math, threading, re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import faiss
from colorama import Style, init; init(autoreset=True)
from extensions_support import load_dataset

DATA_FILE = "data/team-notesv3.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K_SEMANTIC = 50
TOP_K_FINAL = 5
NEON = (5,217,255)
PINK = (255,105,180)
STATE_FILE = ".v8_1_hybrid_state.json"
LOG_FILE = "v8_1_hybrid_runs.csv"
EXACT_MATCH_BOOST = 1.25
CONFIDENCE_THRESHOLD = 0.45
STABILITY_THRESHOLD = 0.60

rgb = lambda r,g,b: f"\033[38;2;{r};{g};{b}m"; RESET = Style.RESET_ALL
def neon(txt): return f"{rgb(*NEON)}{txt}{RESET}"
def pink(txt): return f"{rgb(*PINK)}{txt}{RESET}"
def bar(s,w=22): f=max(0,min(1,float(s))); n=int(w*f); return f"{rgb(*NEON)}"+("█"*n+"░"*(w-n))+RESET
def badge(s): return "🧩 STABLE" if s>=0.80 else "⚖️ MIXED" if s>=0.55 else "🌪️ CHAOTIC"

def normalize_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

def keyword_score(query, document):
    query_norm = normalize_text(query)
    doc_norm = normalize_text(document)
    stop_words = {'the', 'and', 'for', 'are', 'who', 'what', 'where', 'how', 'you', 'not', 'is', 'to', 'do', 'i', 'it'}
    query_words = [w for w in query_norm.split() if len(w) >= 3 and w not in stop_words]
    if not query_words:
        return 0.0
    matches = sum(1 for word in query_words if word in doc_norm or word[:-1] in doc_norm if word.endswith('s') or word + 's' in doc_norm)
    return matches / len(query_words)

def has_match(query, docs):
    return any(keyword_score(query, d) > 0 for d in docs)

class StreamingIndex:
    def __init__(self, df, text_col):
        self.df = df
        self.text_col = text_col
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.embeddings = None
        self.all_texts = df[text_col].astype(str).tolist()

    def build(self):
        texts = self.df[self.text_col].astype(str).tolist()
        self.embeddings = self.embedder.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype(np.float32))

    def hybrid_search(self, qvec, query_text, k_semantic=50, k_final=5):
        if qvec.ndim == 1: qvec = qvec[None, :]
        faiss.normalize_L2(qvec)
        D, I = self.index.search(qvec.astype(np.float32), k_semantic)
        
        candidates = []
        for dist, idx in zip(D[0], I[0]):
            text = self.all_texts[idx]
            kw = keyword_score(query_text, text)
            combined = 0.6 * float(dist) + 0.4 * kw
            candidates.append({'idx': int(idx), 'sem': float(dist), 'kw': kw, 'combo': combined, 'text': text})
        
        candidates.sort(key=lambda x: x['combo'], reverse=True)
        top = candidates[:k_final]
        return np.array([t['sem'] for t in top]), np.array([t['idx'] for t in top]), top

def softmax(x, temp=1.0):
    x = np.asarray(x, dtype=np.float32) / max(1e-6, float(temp))
    x = x - x.max()
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-9)

def stasis_metrics(qvec, nbh, sims, temp):
    w = softmax(sims, temp)
    centroid = (w[:,None] * nbh).sum(axis=0)
    norm = np.linalg.norm(centroid) + 1e-9
    centroid = centroid / norm
    to_c = nbh @ centroid
    return float((w * to_c).sum()), float((w * sims).sum()), float(np.var(sims)), w

def autotune(qvec, nbh, sims):
    temp, low, high = 0.85, 0.20, 2.50
    for _ in range(8):
        stab, conf, var, w = stasis_metrics(qvec, nbh, sims, temp)
        if abs(stab - 0.80) <= 0.05: break
        if stab < 0.80: high = temp; temp = (temp + low) / 2
        else: low = temp; temp = (temp + high) / 2
    return stab, conf, var, temp, w

def highlight(text, keywords):
    out = text
    for kw in keywords:
        if len(kw) < 3: continue
        out = re.sub(re.escape(kw), lambda m: pink(m.group()), out, flags=re.IGNORECASE)
    return out

def main():
    df, text_col, _ = load_dataset(DATA_FILE)
    print("🔢 Building FAISS index...")
    idx = StreamingIndex(df, text_col)
    idx.build()
    print(f"✅ Ready ({len(df)} docs)")
    
    print(f"\n{neon('🤖 Stasis v8.1-HYBRID (Semantic + Keyword)')}")
    print("Type 'quit' to exit\n")
    
    embedder = idx.embedder
    
    while True:
        try:
            q = input(neon("💬 You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == 'quit': break
        
        qvec = embedder.encode([q], convert_to_numpy=True)[0].astype(np.float32)
        sims, ids, candidates = idx.hybrid_search(qvec, q)
        nbh = idx.embeddings[ids]
        
        base_stab, base_conf, base_var, _ = stasis_metrics(qvec, nbh, sims, 1.0)
        tuned_stab, tuned_conf, tuned_var, temp, _ = autotune(qvec, nbh, sims)
        
        docs = [c['text'] for c in candidates]
        did_boost = has_match(q, docs)
        boosted = min(tuned_conf * EXACT_MATCH_BOOST, 0.95) if did_boost else tuned_conf
        
        print(f"\nBaseline: Stab[{base_stab:.3f}] Conf[{base_conf:.3f}]")
        print(f"Auto-tuned: Stab[{tuned_stab:.3f}] Conf[{tuned_conf:.3f}]")
        if did_boost: print(f"{neon('🚀 BOOST:')} {tuned_conf:.3f} → {boosted:.3f}")
        
        q_terms = [t for t in q.lower().split() if len(t) >= 3]
        print(neon("\n🔍 Evidence:"))
        for i, c in enumerate(candidates, 1):
            line = highlight(c['text'][:200], q_terms)
            kw_tag = neon(f"[KEYWORD {int(c['kw']*100)}%]") if c['kw'] > 0 else ""
            print(f"#{i} {line}... {kw_tag}")
        print("-" * 50)
        
        if boosted >= CONFIDENCE_THRESHOLD and tuned_stab >= STABILITY_THRESHOLD:
            snippets = []
            for c in candidates:
                txt = c['text']
                if "Name:" in txt and "Role:" in txt:
                    name = role = ""
                    for ln in txt.split('\n'):
                        if ln.startswith("Name:"): name = ln.replace("Name:", "").strip()
                        elif ln.startswith("Role:"): role = ln.replace("Role:", "").strip()
                    if name and role: snippets.append(f"{name} - {role}")
                else:
                    for seg in [s.strip() for s in txt.replace('?','.').split('.')]:
                        if any(t in seg.lower() for t in q_terms) and 6 <= len(seg.split()) <= 24:
                            snippets.append(seg); break
                if len(snippets) >= 3: break
            print(f"\nStasis: {'; '.join(snippets)}")
        else:
            print("\nStasis: Insufficient evidence (showing above)")
        print()

if __name__ == "__main__":
    main()
