#!/usr/bin/env python3
# Nemotron + Dataset Probability Stasis v8.1 (Streaming + Auto-Tune + Measurement)
# - Streams new rows from chat_messages.csv into FAISS without rebuilds
# - Auto-tunes entropy (softmax temperature) to target stability ~= 0.80 ± 0.05
# - Measures baseline vs improved stability/confidence and prints deltas
# - Neon (#05D9FF) UI + pink evidence highlights
# - Deterministic, no extra HF model loads (only sentence-transformers + faiss-cpu)

import os, sys, time, json, math, threading
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

# deps: sentence-transformers, faiss-cpu, colorama
from sentence_transformers import SentenceTransformer, util
import faiss
from colorama import Style, init; init(autoreset=True)

# ------------------- Config -------------------
DATA_FILE = "chat_messages.csv"
TEXT_COL_CANDIDATES = ["body_full", "body", "text", "message"]
USER_COL_CANDIDATES = ["login", "user", "username", "author"]
MODEL_NAME = "all-MiniLM-L6-v2"      # fast & light
EMBED_DIM = 384                      # for all-MiniLM-L6-v2
TOP_K = 20                           # neighborhood for stasis
TARGET_STABILITY = 0.80              # auto-tune goal
TARGET_TOL = 0.05
AUTOTUNE_STEPS = 8
AUTOTUNE_INIT_TEMP = 0.85
NEON = (5,217,255)                   # #05D9FF
PINK = (255,105,180)                 # hot pink-ish for highlights
STATE_FILE = ".v8_1_stream_state.json"
LOG_FILE = "v8_1_runs.csv"

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

# ------------------- Data / Index -------------------
class StreamingIndex:
    def __init__(self, df: pd.DataFrame, text_col: str):
        self.df = df
        self.text_col = text_col
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatIP(EMBED_DIM)   # cosine via normalized vectors
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
        # new chunk
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
        # expects normalized qvec
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
    # entropy-weighted centroid and stability
    w = softmax(sims, temp=temp)   # smaller temp -> peakier -> higher stability
    centroid = (w[:,None] * nbh_vecs).sum(axis=0)
    # normalize centroid for cosine with members
    norm = np.linalg.norm(centroid) + 1e-9
    centroid = centroid / norm
    # stability: weighted mean similarity to centroid
    to_centroid = (nbh_vecs @ centroid)
    stability = float((w * to_centroid).sum())
    confidence = float((w * sims).sum())  # how similar the neighbors are to query
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
        # if stability too low -> decrease temp (sharper weights); too high -> increase temp
        if stab < target: high = temp; temp = (temp + low)/2
        else:             low  = temp; temp = (temp + high)/2
    return best  # (stability, confidence, variance, temp, weights)

def highlight_evidence(text: str, keywords: List[str]) -> str:
    out = text
    for kw in sorted(set([k for k in keywords if k]), key=len, reverse=True):
        out = out.replace(kw, pink(kw))
        # also case-insensitive simple pass
        out = out.replace(kw.capitalize(), pink(kw.capitalize()))
        out = out.replace(kw.upper(), pink(kw.upper()))
    return out

# ------------------- Load dataset & columns -------------------
def load_dataset():
    if not Path(DATA_FILE).exists():
        print(f"{neon('⚠️')} {DATA_FILE} not found. Place it in this directory.")
        sys.exit(1)
    df = pd.read_csv(DATA_FILE)
    text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    if text_col is None:
        # fallback: pick the longest-text column
        text_col = max(df.columns, key=lambda c: df[c].astype(str).map(len).mean())
    user_col = next((c for c in USER_COL_CANDIDATES if c in df.columns), None)
    print(f"🧾 Using text column: {text_col}" + (f" | user column: {user_col}" if user_col else ""))
    df[text_col] = df[text_col].astype(str).fillna("")
    return df, text_col, user_col

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
              "tuned_stability","tuned_conf","delta_stability","delta_conf","temp","top_terms"]
    exists = Path(LOG_FILE).exists()
    import csv, datetime
    row_out = {
        "ts": datetime.datetime.utcnow().isoformat(),
        **row
    }
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow({k: row_out.get(k,"") for k in header})

# ------------------- Main interactive -------------------
def main():
    df, text_col, user_col = load_dataset()
    print("🔢 Building (or updating) FAISS index...")
    idx = StreamingIndex(df, text_col)
    idx.build()
    print(f"✅ FAISS index ready ({len(df)} rows).")

    state = read_state()
    last_mtime = Path(DATA_FILE).stat().st_mtime

    embedder = idx.embedder  # reuse
    print(f"\n{neon('🤖 Nemotron + Dataset Probability Stasis v8.1 (Streaming + Auto-Tune)')}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            q = input(neon("💬 You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q: continue
        if q.lower() in {"quit","exit"}: break

        # streaming check
        mtime = Path(DATA_FILE).stat().st_mtime
        if mtime > last_mtime:
            # reload only new rows
            new_df = pd.read_csv(DATA_FILE)
            new_rows = len(new_df) - len(df)
            df = new_df
            idx.df = df
            added = idx.maybe_stream_new_rows()
            last_mtime = mtime
            print(neon(f"🔁 Detected dataset growth: +{added} rows streamed into FAISS."))

        # encode query
        qvec = embedder.encode([q], convert_to_numpy=True)[0].astype(np.float32)
        faiss.normalize_L2(qvec.reshape(1,-1))

        sims, ids = idx.knn(qvec, TOP_K)
        nbh_vecs = idx.embeddings[ids]

        # -------- Baseline (no auto-tune; fixed temp=1.0) --------
        base_stab, base_conf, base_var, base_w = stasis_metrics(qvec, nbh_vecs, sims, temp=1.0)

        # -------- Auto-tune to target --------
        tuned_stab, tuned_conf, tuned_var, tuned_temp, tuned_w = autotune_temperature(qvec, nbh_vecs, sims,
                                                                                      init_temp=AUTOTUNE_INIT_TEMP,
                                                                                      target=TARGET_STABILITY,
                                                                                      tol=TARGET_TOL)
        # -------- Measure improvement --------
        d_stab = tuned_stab - base_stab
        d_conf = tuned_conf - base_conf

        print()
        print_metrics("Baseline", base_stab, base_conf, extra=f"Variance [{base_var:.3f}]")
        print()
        print_metrics("Improved (Auto-tuned)", tuned_stab, tuned_conf,
                      extra=f"Variance [{tuned_var:.3f}] | Temp [{tuned_temp:.3f}] | ΔStab [{d_stab:+.3f}] | ΔConf [{d_conf:+.3f}]")
        print()

        # -------- Evidence terms & print excerpts --------
        # choose ~5 highest-weighted neighbors; extract top tokens by simple tf-ish split
        topN = min(5, len(ids))
        chosen = list(zip(ids[:topN], tuned_w[:topN], sims[:topN]))
        # simple keyword extraction: top frequent words across chosen lines (lowercased) intersecting with query terms
        lines = [df[text_col].iloc[i] for i,_w,_s in chosen]
        # naive keywords: tokens that appear in query or are frequent in lines
        q_terms = set([t for t in q.lower().split() if len(t)>=3])
        token_counts = {}
        for ln in lines:
            for tok in ln.lower().split():
                tok = ''.join(ch for ch in tok if ch.isalnum())
                if len(tok)<3: continue
                token_counts[tok] = token_counts.get(tok,0)+1
        top_tokens = sorted(token_counts.items(), key=lambda x:x[1], reverse=True)[:6]
        keywords = list(q_terms.union({t for t,_c in top_tokens}))

        print(neon("🔍 Top dataset evidence (auto-tuned):"))
        for rank,(i,w,s) in enumerate(chosen, start=1):
            raw = df[text_col].iloc[i]
            line = highlight_evidence(raw, keywords)
            print(f"#{rank} {line}")
        print("-"*56)

        # -------- Compose a safe answer from evidence --------
        # If tuned_conf and tuned_stab are decent, produce a short synthesis from top lines
        if tuned_conf >= 0.45 and tuned_stab >= 0.60:
            # heuristic: extract a few most relevant sentences from top lines
            snippets = []
            for ln in lines:
                # keep short phrases that include any q term
                kept = []
                for seg in [s.strip() for s in ln.replace("?",".").replace("!"," .").split(".")]:
                    if any(t in seg.lower() for t in q_terms) or len(q_terms)==0:
                        if 6 <= len(seg.split()) <= 24:
                            kept.append(seg)
                if kept:
                    snippets.append(kept[0])
                if len(snippets) >= 3: break
            synthesis = " ".join(snippets) if snippets else None
        else:
            synthesis = None

        print()
        if synthesis:
            print("╭" + "─"*46 + "╮")
            print("Nemotron says:")
            print(synthesis)
            print("╰" + "─"*46 + "╯")
        else:
            print("╭" + "─"*46 + "╮")
            print("Nemotron says:")
            print("Insufficient stable evidence for a confident synthesis. Showing evidence above.")
            print("╰" + "─"*46 + "╯")

        # -------- Persist run metrics --------
        log_run({
            "query": q,
            "baseline_stability": f"{base_stab:.4f}",
            "baseline_conf": f"{base_conf:.4f}",
            "tuned_stability": f"{tuned_stab:.4f}",
            "tuned_conf": f"{tuned_conf:.4f}",
            "delta_stability": f"{d_stab:.4f}",
            "delta_conf": f"{d_conf:.4f}",
            "temp": f"{tuned_temp:.4f}",
            "top_terms": ",".join(keywords[:8])
        })
        print()

if __name__ == "__main__":
    main()
