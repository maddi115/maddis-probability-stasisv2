#!/usr/bin/env python3
# Nemotron + Dataset Probability Stasis v8  — Probability Field Map
# - Entropy-weighted stasis (less weight for divergent points)
# - Fast: FAISS variance patterns, no multi-LLM loops
# - Visual: 2D field map (pink=stable, blue=chaotic); ASCII fallback
# - Safe generation: LLM composes ONLY from the high-stability cluster
#
# Run:
#   python3 withdatasetv8_probabilityfieldmap.py
#   (then type your question)
# Options:
#   STASIS_TOPK (env, default 64)   — neighbors to consider
#   STASIS_COL  (env, default body_full) — text column to index
#   STASIS_USER (env, default login)     — user/author column
#   STASIS_REBUILD=1 to force re-embed/reindex

import os, sys, json, time, math, re, hashlib, pathlib
from pathlib import Path
import numpy as np
import pandas as pd

# Light deps (already in your env in earlier steps)
import faiss
from sentence_transformers import SentenceTransformer, util

# Optional: matplotlib for a pretty scatter; otherwise ASCII fallback
try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# Optional LLM (llama.cpp gguf) — only used AFTER stasis picks evidence
try:
    from llama_cpp import Llama
    _HAVE_LLM = True
except Exception:
    _HAVE_LLM = False

# ---------- Config ----------
DATA_FILE   = "chat_messages.csv"
TEXT_COL    = os.getenv("STASIS_COL",  "body_full")
USER_COL    = os.getenv("STASIS_USER", "login")
TOPK        = int(os.getenv("STASIS_TOPK", "64"))
CACHE_DIR   = Path(".stasis_cache_v8")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMB_MODEL   = "all-MiniLM-L6-v2"     # 384-dim
LLM_MODEL   = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
CTX, LAYERS, SEED = 4096, 40, 42
PINK = (255, 121, 198)   # #FF79C6
BLUE = (90, 200, 255)    # soft-neon blue
NEON = (5, 217, 255)     # #05D9FF for UI accents

def rgb(r,g,b): return f"\033[38;2;{r};{g};{b}m"
RESET = "\033[0m"

def neon(txt): return f"{rgb(*NEON)}{txt}{RESET}"
def bar(s,w=22):
    s = max(0,min(1,float(s)))
    fill = int(s*w)
    return f"{rgb(*NEON)}"+("█"*fill+"░"*(w-fill))+RESET
def badge(s):
    return "🧩 STABLE" if s>=0.8 else "⚖️ MIXED" if s>=0.6 else "🌪️ CHAOTIC"

def pink_blue(val):
    # val in [0,1] -> blend PINK -> BLUE
    val = max(0,min(1,val))
    r = int(PINK[0]*(val) + BLUE[0]*(1-val))
    g = int(PINK[1]*(val) + BLUE[1]*(1-val))
    b = int(PINK[2]*(val) + BLUE[2]*(1-val))
    return rgb(r,g,b)

def short(s, n=140):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return (s[:n-1] + "…") if len(s) > n else s

def hash_for_cache(path, text_col):
    st = os.stat(path)
    m = hashlib.sha1()
    m.update(f"{path}:{st.st_mtime_ns}:{text_col}".encode())
    return m.hexdigest()[:12]

# ---------- Load dataset ----------
if not Path(DATA_FILE).exists():
    print(f"{neon('⚠️')} {DATA_FILE} not found. Place it next to this script.")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
if TEXT_COL not in df.columns:
    print(f"{neon('⚠️')} '{TEXT_COL}' column not found. Available: {list(df.columns)}")
    sys.exit(1)
if USER_COL not in df.columns:
    print(f"{neon('⚠️')} '{USER_COL}' column not found. Available: {list(df.columns)}")
    sys.exit(1)

texts = df[TEXT_COL].fillna("").astype(str).tolist()
users = df[USER_COL].fillna("").astype(str).tolist()
N = len(texts)

print(neon("📦 Loading dataset..."))
print(f"🧾 Using text column: {TEXT_COL} | user column: {USER_COL} ({N} rows)")

# ---------- Build / Load FAISS + embeddings ----------
cache_id = hash_for_cache(DATA_FILE, TEXT_COL)
E_PATH    = CACHE_DIR / f"emb_{cache_id}.npy"
IDX_PATH  = CACHE_DIR / f"faiss_{cache_id}.index"
USR_PATH  = CACHE_DIR / f"users_{cache_id}.npy"

rebuild = os.getenv("STASIS_REBUILD", "0") == "1"
embedder = SentenceTransformer(EMB_MODEL)

if (not E_PATH.exists()) or (not IDX_PATH.exists()) or rebuild:
    print(neon("🔢 Encoding dataset (Probability Field build)..."))
    # Batch encode
    B=256
    embs = []
    for i in range(0, N, B):
        embs.append(embedder.encode(texts[i:i+B], convert_to_numpy=True, normalize_embeddings=True))
    X = np.vstack(embs).astype("float32")  # (N, d)
    np.save(E_PATH, X)
    np.save(USR_PATH, np.array(users, dtype=object))
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    faiss.write_index(index, str(IDX_PATH))
    print(neon(f"✅ FAISS index built ({N} vectors)."))
else:
    X = np.load(E_PATH)
    users_np = np.load(USR_PATH, allow_pickle=True)
    index = faiss.read_index(str(IDX_PATH))
    print(neon(f"✅ FAISS index cached ({N} vectors)."))

# ---------- ASCII heatmap fallback ----------
def ascii_heatmap(points, weights, width=42, height=16):
    if len(points)==0:
        return "(no points)"
    pts = np.asarray(points)  # (k,2)
    wts = np.asarray(weights).astype(float)
    # Normalize to [0,1] box
    mins = pts.min(0); maxs = pts.max(0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (pts - mins) / span
    canvas = np.zeros((height, width), dtype=float)
    for (x,y), wt in zip(norm, wts):
        xi = min(width-1, max(0, int(x*(width-1))))
        yi = min(height-1, max(0, int((1-y)*(height-1))))
        canvas[yi, xi] += wt
    # normalize per-cell
    if canvas.max()>0: canvas = canvas / canvas.max()
    # map to chars
    CH = " .:-=+*#%@"
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            v = canvas[r,c]
            idx = min(len(CH)-1, int(v*(len(CH)-1)))
            row.append(CH[idx])
        out.append("".join(row))
    return "\n".join(out)

# ---------- tiny sentiment lexicon (no transformers) ----------
_POS = set("love like enjoy awesome great good tasty delicious amazing favorite fav yum yummy crave craving".split())
_NEG = set("hate dislike gross bad awful nasty terrible meh sick tired sad disappointed".split())

def tiny_sentiment(s: str) -> float:
    # returns [-1,1]
    toks = re.findall(r"[a-zA-Z']+", s.lower())
    if not toks: return 0.0
    pos = sum(t in _POS for t in toks)
    neg = sum(t in _NEG for t in toks)
    total = pos+neg
    if total==0: return 0.0
    return (pos - neg) / total

# ---------- Entropy-weighted stasis ----------
def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    return (arr - arr.mean()) / (arr.std() + 1e-6)

def entropy_weights(sims):
    # lower weight for outliers via similarity z-score
    z = zscore(sims)
    # convert to [0,1] weight via sigmoid; negative z (divergent) -> lower weight
    w = 1/(1+np.exp(-z))
    return w

def cluster_pick(query_vec, neigh_vecs, sims, k=2):
    # lightweight 2-means over neighborhood; choose cluster with higher mean sim
    from sklearn.cluster import KMeans
    K = min(k, len(neigh_vecs))
    if K<=1: return np.arange(len(neigh_vecs)), np.ones(len(neigh_vecs))
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(neigh_vecs)
    labels = km.labels_
    best, best_mean = None, -9
    for lab in range(K):
        idx = np.where(labels==lab)[0]
        m = sims[idx].mean() if len(idx)>0 else -9
        if m>best_mean:
            best, best_mean = idx, m
    return best, labels

def stasis_metrics(sims, weights, sentiments):
    sims = np.asarray(sims, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum()<=0: weights = np.ones_like(weights)
    mean_sim = float((sims*weights).sum()/weights.sum())
    var_sim  = float(((sims-mean_sim)**2 * weights).sum()/weights.sum())
    # stability ↑ with higher mean and lower variance
    stability = float(max(0.0, min(1.0, 0.5*mean_sim + 0.5*(1.0 - min(1.0, var_sim)))))
    # confidence just normalized mean similarity
    confidence = float(max(0.0, min(1.0, mean_sim)))
    # sentiment variance (lower is better)
    s_var = float(np.var(sentiments)) if len(sentiments)>0 else 0.0
    return stability, confidence, s_var

# ---------- Optional LLM (compose from evidence only) ----------
_LLM = None
def get_llm():
    global _LLM
    if _LLM is None and _HAVE_LLM:
        _LLM = Llama(model_path=LLM_MODEL, n_ctx=CTX, n_gpu_layers=LAYERS, seed=SEED, verbose=False)
    return _LLM

SYS_PROMPT = (
"You are Nemotron, a factual reasoning model.\n"
"Use ONLY the provided evidence lines. Do not reveal internal notes.\n"
"Answer in ≤3 sentences, be concise and concrete.\n"
)

def generate_answer(question, evidence_lines):
    if not _HAVE_LLM or len(evidence_lines)==0:
        # fallback: simple extractive summary
        joined = " | ".join(evidence_lines)[:400]
        return f"(No LLM) Based on evidence: {joined}"
    llm = get_llm()
    prompt = f"{SYS_PROMPT}\nQuestion: {question}\nEvidence:\n" + "\n".join(f"- {e}" for e in evidence_lines) + "\nAnswer:"
    out = llm(prompt, temperature=0.6, max_tokens=160, top_p=0.9, stop=["Question:", "Evidence:"])
    txt = out["choices"][0]["text"].strip()
    return txt

# ---------- Main loop ----------
print(neon("\n🤖 Nemotron + Dataset Probability Stasis v8 (Field Map)"))
print("Type 'quit' to exit.\n")

while True:
    try:
        q = input(f"{rgb(*NEON)}💬 You:{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not q or q.lower() in {"quit","exit"}:
        break

    # 1) FAISS retrieve
    qv = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, min(TOPK, N))  # inner product == cosine because normalized
    sims = D[0]; idxs = I[0]
    neigh_vecs = X[idxs]
    lines = [texts[i] for i in idxs]
    neigh_users = [users[i] for i in idxs]

    # 2) Entropy weights + tiny sentiment
    w_ent = entropy_weights(sims)
    sents = np.array([tiny_sentiment(t) for t in lines], dtype=float)
    # combine weights: similarity * entropy * (1 - sentiment variance boost)
    w = (np.maximum(0,sims) * w_ent)

    # 3) Pick stable cluster
    keep_idx, labels = cluster_pick(qv[0], neigh_vecs, sims, k=2)
    keep_idx = np.asarray(keep_idx)
    sims_k = sims[keep_idx]
    w_k    = w[keep_idx]
    sents_k= sents[keep_idx]
    lines_k= [lines[i] for i in keep_idx]
    users_k= [neigh_users[i] for i in keep_idx]
    vecs_k = neigh_vecs[keep_idx]

    # 4) Metrics
    stability, confidence, s_var = stasis_metrics(sims_k, w_k, sents_k)

    # Save metrics and compare to last
    MET_PATH = CACHE_DIR / "stasis_metrics_last.json"
    last = {}
    if MET_PATH.exists():
        try: last = json.loads(MET_PATH.read_text())
        except: last = {}
    curr = {"stability": stability, "confidence": confidence, "sent_var": s_var, "ts": time.time()}
    MET_PATH.write_text(json.dumps(curr, indent=2))

    def delta_str(k):
        if k not in last: return ""
        d = curr[k]-last[k]
        arrow = "↑" if d>1e-6 else ("↓" if d<-1e-6 else "→")
        return f" {arrow} {d:+.3f}"

    # 5) Visual field map (TSNE for neighborhood)
    #    Map top neighbors into 2D and color by normalized weight
    have_points = False
    pts2 = None; weights_n = None
    if len(vecs_k) >= 3:
        try:
            from sklearn.manifold import TSNE
            ts = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=min(30, max(5, len(vecs_k)//3)), random_state=SEED)
            pts2 = ts.fit_transform(vecs_k)
            wmin, wmax = float(w_k.min()), float(w_k.max())+1e-9
            weights_n = (w_k - wmin)/(wmax - wmin)
            have_points = True
        except Exception:
            have_points = False

    # 6) Compose answer strictly from evidence in stable cluster
    # pick top few evidences by weight
    top_order = np.argsort(-w_k)[:5]
    evidence = [short(lines_k[i], 220) for i in top_order]
    answer = generate_answer(q, evidence)

    # ---------- OUTPUT ----------
    print()
    print(f"{neon('🏆 Stability')} [{stability:.3f}] {bar(stability)} {badge(stability)}")
    print(f"📊 Dataset Confidence [{confidence:.3f}]{delta_str('confidence')}")
    print(f"🧪 Sentiment variance [{s_var:.3f}]{delta_str('sent_var')}")
    print(f"🧭 Stability change vs last run:{delta_str('stability') or ' (no baseline)'}")
    print()

    # Pretty box
    print("╭" + "─"*46 + "╮")
    print("Nemotron says:")
    for line in re.findall(r'.{1,80}(?:\s+|$)', answer):
        print(line.rstrip())
    print("╰" + "─"*46 + "╯")

    print("\n🔍 Top dataset evidence (by stability weight):")
    for rank, i in enumerate(top_order, 1):
        wv = float(w_k[i])
        txt = lines_k[i]
        # simple pink highlight for query terms present
        hi = txt
        for tok in set(re.findall(r"[a-zA-Z0-9']+", q.lower())):
            if not tok or len(tok)<3: continue
            hi = re.sub(rf"(?i)\b({re.escape(tok)})\b", pink_blue(0.85)+r"\1"+RESET, hi)
        print(f"#{rank} {pink_blue(min(1, wv / (w_k.max()+1e-9)))}{short(hi, 200)}{RESET}")

    # Visual map
    if have_points and _HAVE_MPL:
        try:
            plt.figure(figsize=(5.0,4.2), dpi=140)
            for (x,y), wn in zip(pts2, weights_n):
                c = pink_blue(wn)
                plt.scatter([x],[y], s=28+wn*40, c=[c.replace("\033[38;2;","").replace("m","")], marker='o')  # color escape won't work here; let mpl pick defaults
            plt.title("Probability Field Map (stable = pink, divergent = blue)")
            plt.xticks([]); plt.yticks([])
            outp = Path("stasis_field.png")
            plt.tight_layout()
            plt.savefig(outp)
            plt.close()
            print(f"\n🗺️  Saved field map image: {outp}")
        except Exception:
            have_points = False

    if have_points and not _HAVE_MPL:
        # ASCII fallback using the TSNE points
        try:
            print("\n🗺️  ASCII Probability Field (pink=stable, blue=chaotic):")
            # normalize weights_n already 0..1
            print(ascii_heatmap(pts2, weights_n))
        except Exception:
            pass

    print("-"*60)

