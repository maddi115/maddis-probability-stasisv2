#!/usr/bin/env python3
# Stasis v8.1-CLEAN (Hybrid Search with Digestible Output)

import os, sys, time, json, re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import faiss
from colorama import Style, init; init(autoreset=True)
from extensions_support import load_dataset

# Config
DATA_FILE = "data/team-notesv3.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
NEON = (5,217,255)
GREEN = (50,205,50)
YELLOW = (255,193,7)
RED = (255,71,87)
STATE_FILE = ".v8_1_clean_state.json"

rgb = lambda r,g,b: f"\033[38;2;{r};{g};{b}m"
RESET = Style.RESET_ALL

def color(txt, rgb_tuple): 
    return f"{rgb(*rgb_tuple)}{txt}{RESET}"

def conf_color(conf):
    if conf >= 0.55: return GREEN
    elif conf >= 0.45: return YELLOW
    else: return RED

def normalize_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

def keyword_score(query, document):
    query_norm = normalize_text(query)
    doc_norm = normalize_text(document)
    stop_words = {'the', 'and', 'for', 'are', 'who', 'what', 'where', 'how', 'you', 'not', 'is', 'to', 'do', 'i', 'it'}
    query_words = [w for w in query_norm.split() if len(w) >= 3 and w not in stop_words]
    if not query_words: return 0.0
    matches = sum(1 for word in query_words if word in doc_norm or (word.endswith('s') and word[:-1] in doc_norm))
    return matches / len(query_words)

def has_match(query, docs):
    return any(keyword_score(query, d) > 0 for d in docs)

class StreamingIndex:
    def __init__(self, df, text_col):
        self.df = df
        self.text_col = text_col
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
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
            candidates.append({
                'idx': int(idx), 
                'sem': float(dist), 
                'kw': kw, 
                'combo': combined, 
                'text': text
            })
        
        candidates.sort(key=lambda x: x['combo'], reverse=True)
        return np.array([t['sem'] for t in candidates[:k_final]]), candidates[:k_final]

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
    stab = float((w * to_c).sum())
    conf = float((w * sims).sum())
    return stab, conf

def autotune(qvec, nbh, sims):
    temp = 0.85
    for _ in range(8):
        stab, conf = stasis_metrics(qvec, nbh, sims, temp)
        if abs(stab - 0.80) <= 0.05: break
        if stab < 0.80: temp = (temp + 0.20) / 2
        else: temp = (temp + 2.50) / 2
    return stab, conf, temp

def format_answer(candidates, query_terms):
    """Extract clean answer from candidates"""
    snippets = []
    for c in candidates:
        txt = c['text']
        if "Name:" in txt and "Role:" in txt:
            name = role = ""
            for ln in txt.split('\n'):
                if ln.startswith("Name:"): name = ln.replace("Name:", "").strip()
                elif ln.startswith("Role:"): role = ln.replace("Role:", "").strip()
            if name and role: snippets.append(f"{name} → {role}")
        else:
            for seg in [s.strip() for s in txt.replace('?','.').split('.')]:
                if any(t in seg.lower() for t in query_terms) and 6 <= len(seg.split()) <= 24:
                    snippets.append(seg)
                    break
        if len(snippets) >= 3: break
    
    return "; ".join(snippets) if snippets else None

def draw_box(content, width=60):
    """Draw a clean box around content"""
    lines = content.split('\n') if isinstance(content, str) else [str(content)]
    print(f"┌{'─' * width}┐")
    for line in lines[:4]:  # Limit to 4 lines
        truncated = line[:width-2] + ".." if len(line) > width-2 else line
        padding = width - len(truncated)
        print(f"│ {truncated}{' ' * padding}│")
    print(f"└{'─' * width}┘")

def main():
    df, text_col, _ = load_dataset(DATA_FILE)
    idx = StreamingIndex(df, text_col)
    idx.build()
    
    print(f"\n{color('🤖 Stasis v8.1-CLEAN', NEON)} | {len(df)} docs loaded")
    print(f"{'─' * 70}")
    print(f"{'Query':<35} {'Conf':>8} {'Stab':>8} {'Status':>12}")
    print(f"{'─' * 70}")

    embedder = idx.embedder
    
    while True:
        try:
            q = input(f"\n{color('❯', NEON)} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == 'quit': break

        qvec = embedder.encode([q], convert_to_numpy=True)[0].astype(np.float32)
        sims, candidates = idx.hybrid_search(qvec, q)
        nbh = idx.embeddings[[c['idx'] for c in candidates]]
        
        stab, conf, temp = autotune(qvec, nbh, sims)
        
        # Boost if keyword match
        if has_match(q, [c['text'] for c in candidates]):
            conf = min(conf * 1.25, 0.95)
        
        query_terms = [t for t in q.lower().split() if len(t) >= 3]
        answer = format_answer(candidates, query_terms)
        
        # Format ratings
        conf_str = color(f"{conf:.2f}", conf_color(conf))
        stab_str = f"{stab:.2f}"
        status = color("✓ ANSWER", GREEN) if answer else color("✗ PASS", RED)
        
        # Print compact row
        q_short = q[:32] + ".." if len(q) > 34 else q
        print(f"{q_short:<35} {conf_str:>10} {stab_str:>8} {status:>14}")
        
        # If there's an answer, show it in a box below
        if answer:
            print(f"\n   {color('→', NEON)} {answer[:120]}{'...' if len(answer) > 120 else ''}")
        
        # Show top evidence on request (optional)
        if "--verbose" in sys.argv:
            for i, c in enumerate(candidates[:3], 1):
                kw_tag = f"[{int(c['kw']*100)}%]" if c['kw'] > 0 else ""
                print(f"   {i}. {c['text'][:60]}... {color(kw_tag, (100,100,100))}")

if __name__ == "__main__":
    main()
