#!/usr/bin/env python3
# Pure Probability Stasis v8.1-EXACT (Forgiving Exact Match Edition)
# - Case-insensitive, punctuation-insensitive, substring matching

import os, sys, time, json, math, threading, re, string
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util
import faiss
from colorama import Style, init; init(autoreset=True)
from extensions_support import load_dataset

# ------------------- Config -------------------
DATA_FILE = "data/team-notesv3.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 20
TARGET_STABILITY = 0.80
TARGET_TOL = 0.05
AUTOTUNE_STEPS = 8
AUTOTUNE_INIT_TEMP = 0.85
NEON = (5,217,255)
PINK = (255,105,180)
STATE_FILE = ".v8_1_exact_state.json"
LOG_FILE = "v8_1_exact_runs.csv"

EXACT_MATCH_BOOST = 1.25
CONFIDENCE_THRESHOLD = 0.45
STABILITY_THRESHOLD = 0.60

# ------------------- UI helpers -------------------
rgb = lambda r,g,b: f"\033[38;2;{r};{g};{b}m"; RESET = Style.RESET_ALL
def neon(txt): return f"{rgb(*NEON)}{txt}{RESET}"
def pink(txt): return f"{rgb(*PINK)}{txt}{RESET}"
def bar(s,w=22): f=max(0,min(1,float(s))); n=int(w*f); return f"{rgb(*NEON)}"+("█"*n+"░"*(w-n))+RESET
def badge(s): return "🧩 STABLE" if s>=0.80 else "⚖️ MIXED" if s>=0.55 else "🌪️ CHAOTIC"

def print_metrics(title, stability, confidence, extra=None, prefix=""):
    print(f"{prefix}{title}")
    print(f"{prefix}🏆 Stability [{stability:.3f}] {bar(stability)} {badge(stability)}")
    print(f"{prefix}📊 Dataset Confidence [{confidence:.3f}]")
    if extra: print(f"{prefix}{extra}")

# ------------------- Forgiving Exact Match Logic -------------------
def normalize_text(text: str) -> str:
    """Remove punctuation and lowercase"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def has_exact_match(query: str, evidence_docs: List[str]) -> bool:
    """
    Forgiving exact match:
    - Case insensitive
    - Punctuation insensitive
    - Substring matching (so 'tailgate' matches 'tailgates')
    - Minimum 3 char length
    """
    # Normalize query
    query_norm = normalize_text(query)
    q_terms = [w for w in query_norm.split() if len(w) >= 3]
    
    if not q_terms:
        return False
    
    # Normalize all evidence into one string
    evidence_norm = normalize_text(' '.join(evidence_docs))
    
    # Check if any query term appears as substring in evidence
    for term in q_terms:
        # Skip common stop words
        if term in ['the', 'and', 'for', 'are', 'who', 'what', 'where', 'how', 'you', 'not']:
            continue
        if term in evidence_norm:
            return True
    
    return False

def calculate_boosted_confidence(base_conf: float, query: str, evidence_docs: List[str]) -> Tuple[float, bool]:
    """Returns (boosted_confidence, did_boost)"""
    if has_exact_match(query, evidence_docs):
        boosted = min(base_conf * EXACT_MATCH_BOOST, 0.95)
        return boosted, True
    return base_conf, False

# ------------------- Data / Index -------------------
class StreamingIndex:
    def __init__(self, df: pd.DataFrame, text_col: str):
        self.df = df
        self.text_col = text_col
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.embeddings = None
        self._normalize = True
        self.last_row = 0

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        if self._normalize:
            faiss.normalize_L2(vecs)
        return vecs.astype(np.float32)

    def build(self):
        texts = self.df[self.text_col].astype(str).tolist()
        self.embeddings = self._encode_batch(texts)
        self.index.add(self.embeddings)
        self.last_row = len(texts)

    def maybe_stream_new_rows(self):
        total = len(self.df)
        if total <= self.last_row: return 0
        new_texts = self.df[self.text_col].astype(str).iloc[self.last_row:total].tolist()
        if not new_texts: return 0
        new_vecs = self._encode_batch(new_texts)
        self.index.add(new_vecs)
        if self.embeddings is None:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])
        added = len(new_texts)
        self.last_row = total
        return added

    def knn(self, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if qvec.ndim == 1: qvec = qvec[None, :]
        D, I = self.index.search(qvec.astype(np.float32), k)
        return D[0], I[0]

# ------------------- Probability Stasis -------------------
def softmax(x, temp=1.0):
    x = np.asarray(x, dtype=np.float32)
    x = x / max(1e-6, float(temp))
    x = x - x.max()
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-9)

def stasis_metrics(qvec: np.ndarray, nbh_vecs: np.ndarray, sims: np.ndarray, temp: float):
    w = softmax(sims, temp=temp)
    centroid = (w[:,None] * nbh_vecs).sum(axis=0)
    norm = np.linalg.norm(centroid) + 1e-9
    centroid = centroid / norm
    to_centroid = (nbh_vecs @ centroid)
    stability = float((w * to_centroid).sum())
    confidence = float((w * sims).sum())
    variance = float(np.var(sims))
    return stability, confidence, variance, w

def autotune_temperature(qvec, nbh_vecs, sims, init_temp=AUTOTUNE_INIT_TEMP, target=TARGET_STABILITY, tol=TARGET_TOL):
    temp = float(init_temp)
    low, high = 0.20, 2.50
    best = None
    for _ in range(AUTOTUNE_STEPS):
        stab, conf, var, w = stasis_metrics(qvec, nbh_vecs, sims, temp)
        best = (stab, conf, var, temp, w)
        if abs(stab - target) <= tol:
            break
        if stab < target: high = temp; temp = (temp + low)/2
        else:             low  = temp; temp = (temp + high)/2
    return best

def highlight_evidence(text: str, keywords: List[str]) -> str:
    """Case-insensitive highlighting"""
    out = text
    for kw in keywords:
        if len(kw) < 3:
            continue
        # Highlight all case variations
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        out = pattern.sub(lambda m: pink(m.group()), out)
    return out

# ------------------- Persistence -------------------
def read_state():
    if Path(STATE_FILE).exists():
        try: return json.loads(Path(STATE_FILE).read_text())
        except: return {}
    return {}

def write_state(d):
    try: Path(STATE_FILE).write_text(json.dumps(d, indent=2))
    except: pass

def log_run(row: Dict):
    header = ["ts","query","baseline_stability","baseline_conf",
              "tuned_stability","tuned_conf","boosted_conf","did_boost","delta_stability","delta_conf","temp","top_terms"]
    exists = Path(LOG_FILE).exists()
    import csv, datetime
    row_out = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **row
    }
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow({k: row_out.get(k,"") for k in header})

# ------------------- Main interactive -------------------
def main():
    df, text_col, user_col = load_dataset(DATA_FILE)
    print("🔢 Building (or updating) FAISS index...")
    idx = StreamingIndex(df, text_col)
    idx.build()
    print(f"✅ FAISS index ready ({len(df)} rows).")

    state = read_state()
    last_mtime = Path(DATA_FILE).stat().st_mtime

    embedder = idx.embedder
    print(f"\n{neon('🤖 Pure Probability Stasis v8.1-EXACT (Forgiving Match)')}")
    print(f"   Boost: {EXACT_MATCH_BOOST}x | Threshold: {CONFIDENCE_THRESHOLD}")
    print("   Case-insensitive | Punctuation-insensitive | Substring matching")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            q = input(neon("💬 You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q: continue
        if q.lower() in {"quit","exit"}: break

        mtime = Path(DATA_FILE).stat().st_mtime
        if mtime > last_mtime:
            new_df = pd.read_csv(DATA_FILE)
            df = new_df
            idx.df = df
            added = idx.maybe_stream_new_rows()
            last_mtime = mtime
            print(neon(f"🔁 Detected dataset growth: +{added} rows streamed into FAISS."))

        qvec = embedder.encode([q], convert_to_numpy=True)[0].astype(np.float32)
        faiss.normalize_L2(qvec.reshape(1,-1))

        sims, ids = idx.knn(qvec, TOP_K)
        nbh_vecs = idx.embeddings[ids]

        base_stab, base_conf, base_var, base_w = stasis_metrics(qvec, nbh_vecs, sims, temp=1.0)
        tuned_stab, tuned_conf, tuned_var, tuned_temp, tuned_w = autotune_temperature(qvec, nbh_vecs, sims,
                                                                                      init_temp=AUTOTUNE_INIT_TEMP,
                                                                                      target=TARGET_STABILITY,
                                                                                      tol=TARGET_TOL)
        
        # EXACT MATCH BOOST LOGIC
        topN = min(5, len(ids))
        chosen_ids = list(ids[:topN])
        evidence_docs = [df[text_col].iloc[i] for i in chosen_ids]
        boosted_conf, did_boost = calculate_boosted_confidence(tuned_conf, q, evidence_docs)
        
        d_stab = tuned_stab - base_stab
        d_conf = boosted_conf - base_conf

        print()
        print_metrics("Baseline", base_stab, base_conf, extra=f"Variance [{base_var:.3f}]")
        print()
        print_metrics("Auto-tuned", tuned_stab, tuned_conf,
                      extra=f"Variance [{tuned_var:.3f}] | Temp [{tuned_temp:.3f}]")
        if did_boost:
            print(f"   {neon('🚀 FORGIVING MATCH BOOST:')} {tuned_conf:.3f} → {boosted_conf:.3f}")
        print()

        chosen = list(zip(ids[:topN], tuned_w[:topN], sims[:topN]))
        lines = [df[text_col].iloc[i] for i,_w,_s in chosen]
        q_terms = set([t for t in q.lower().split() if len(t)>=3])
        token_counts = {}
        for ln in lines:
            for tok in ln.lower().split():
                tok = ''.join(ch for ch in tok if ch.isalnum())
                if len(tok)<3: continue
                token_counts[tok] = token_counts.get(tok,0)+1
        top_tokens = sorted(token_counts.items(), key=lambda x:x[1], reverse=True)[:6]
        keywords = list(q_terms.union({t for t,_c in top_tokens}))

        print(neon("🔍 Top dataset evidence:"))
        for rank,(i,w,s) in enumerate(chosen, start=1):
            raw = df[text_col].iloc[i]
            line = highlight_evidence(raw, keywords)
            # Check for match using forgiving logic
            norm_raw = normalize_text(raw)
            norm_q = normalize_text(q)
            is_match = any(term in norm_raw for term in norm_q.split() if len(term) >= 3 and term not in ['the', 'and', 'for', 'are'])
            marker = neon(" [MATCH]") if is_match else ""
            print(f"#{rank} {line}{marker}")
        print("-"*56)

        # SYNTHESIS LOGIC
        if boosted_conf >= CONFIDENCE_THRESHOLD and tuned_stab >= STABILITY_THRESHOLD:
            snippets = []
            for ln in lines:
                # Handle structured roster entries specially
                if "Name:" in ln and "Role:" in ln:
                    name = ""
                    role = ""
                    for line in ln.split('\n'):
                        if line.startswith("Name:"):
                            name = line.replace("Name:", "").strip()
                        elif line.startswith("Role:"):
                            role = line.replace("Role:", "").strip()
                    if name and role:
                        snippets.append(f"{name} - {role}")
                    if len(snippets) >= 3: 
                        break
                    continue
                
                # Standard sentence extraction for prose
                kept = []
                for seg in [s.strip() for s in ln.replace("?",".").replace("!"," .").split(".")]:
                    if any(t in seg.lower() for t in q_terms) or len(q_terms)==0:
                        if 6 <= len(seg.split()) <= 24:
                            kept.append(seg)
                if kept:
                    snippets.append(kept[0])
                if len(snippets) >= 3: 
                    break
            
            synthesis = "; ".join(snippets) if snippets else None
        else:
            synthesis = None

        print()
        if synthesis:
            print("╭" + "─"*46 + "╮")
            print("Stasis says:")
            print(synthesis)
            print("╰" + "─"*46 + "╯")
        else:
            print("╭" + "─"*46 + "╮")
            print("Stasis says:")
            print("Insufficient stable evidence for a confident synthesis. Showing evidence above.")
            print("╰" + "─"*46 + "╯")

        log_run({
            "query": q,
            "baseline_stability": f"{base_stab:.4f}",
            "baseline_conf": f"{base_conf:.4f}",
            "tuned_stability": f"{tuned_stab:.4f}",
            "tuned_conf": f"{tuned_conf:.4f}",
            "boosted_conf": f"{boosted_conf:.4f}",
            "did_boost": str(did_boost),
            "delta_stability": f"{d_stab:.4f}",
            "delta_conf": f"{d_conf:.4f}",
            "temp": f"{tuned_temp:.4f}",
            "top_terms": ",".join(keywords[:8])
        })
        print()

if __name__ == "__main__":
    main()
