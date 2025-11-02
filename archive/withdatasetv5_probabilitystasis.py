#!/usr/bin/env python3
# Nemotron + Probability Stasis (Dataset v5, identity-aware)
# - Embeds:  user=<login> :: message=<body_full>
# - FAISS cached index, #05D9FF neon
# - Stability = mean(sim) - 0.5 * std(sim)
# - Retrieval k=15
# - Clean output (no chain-of-thought)

import os, sys, time, json, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd

# Optional deps with friendly errors
try:
    import faiss
except Exception as e:
    print("❌ faiss not installed. Do:  pip install faiss-cpu")
    raise
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    print("❌ sentence-transformers not installed. Do:  pip install sentence-transformers")
    raise
try:
    from llama_cpp import Llama
except Exception:
    print("❌ llama-cpp-python not installed. Do:  pip install llama-cpp-python")
    raise

# =========== Config ===========
DATA_FILE   = "chat_messages.csv"
TEXT_COLS   = ["body_full", "body"]
USER_COLS   = ["login", "user_id"]
MODEL_PATH  = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
CTX         = 4096
LAYERS      = 40
SEED        = 42
TOP_K       = 15                 # retrieved snippets
CANDIDATE_PATHS = [0.62, 0.72, 0.82, 0.9]  # temps
STABILITY_STD_WEIGHT = 0.5       # score = mean - w*std
CACHE_DIR   = Path(".stasis_cache")
EMB_MODEL   = "all-MiniLM-L6-v2"

# Neon #05D9FF
NEON = (5, 217, 255)
def _rgb(r,g,b): return f"\033[38;2;{r};{g};{b}m"
RESET = "\033[0m"
def neon(txt): return f"{_rgb(*NEON)}{txt}{RESET}"
def bar(s,w=22):
    s = max(0.0, min(1.0, float(s)))
    filled = int(round(w*s))
    return _rgb(*NEON) + ("█"*filled + "░"*(w-filled)) + RESET
def badge(s):
    return "🧩 STABLE" if s >= 0.80 else ("⚖️ MIXED" if s >= 0.55 else "🌪️ CHAOTIC")

# =========== Helpers ===========
def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def dataset_signature(path: Path, text_col: str, user_col: str) -> str:
    p = path.resolve()
    stat = p.stat()
    sig = json.dumps({
        "file": str(p),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "text_col": text_col,
        "user_col": user_col,
        "emb_model": EMB_MODEL
    }, sort_keys=True).encode()
    return hashlib.md5(sig).hexdigest()

def build_or_load_index(df: pd.DataFrame, text_col: str, user_col: str):
    CACHE_DIR.mkdir(exist_ok=True)
    sig = dataset_signature(Path(DATA_FILE), text_col, user_col)
    emb_path = CACHE_DIR / f"emb_{sig}.npy"
    meta_path = CACHE_DIR / f"meta_{sig}.json"
    index_path = CACHE_DIR / f"faiss_{sig}.index"

    # Build corpus lines as "user=... :: message=..."
    users = df[user_col].astype(str).fillna("unknown").values
    texts = df[text_col].astype(str).fillna("").values
    corpus = np.array([f"user={u} :: message={t}" for u,t in zip(users, texts)], dtype=object)

    if emb_path.exists() and index_path.exists() and meta_path.exists():
        print("✅ FAISS index cached (identity-aware).")
        embs = np.load(emb_path)
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return corpus, embs, index, meta

    print("🔢 Encoding dataset (identity-aware) + building FAISS ...")
    embedder = SentenceTransformer(EMB_MODEL)
    # batch encode
    B = 256
    vecs = []
    for i in range(0, len(corpus), B):
        chunk = corpus[i:i+B].tolist()
        vec = embedder.encode(chunk, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        vecs.append(vec.astype("float32"))
    embs = np.vstack(vecs)
    d = embs.shape[1]

    index = faiss.IndexFlatIP(d)
    index.add(embs)

    np.save(emb_path, embs)
    faiss.write_index(index, str(index_path))
    meta = {"rows": int(len(corpus)), "dim": int(d), "text_col": text_col, "user_col": user_col}
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"✅ FAISS index built ({meta['rows']} vectors).")
    return corpus, embs, index, meta

def extract_user_tokens(q: str):
    # crude capture of tokens like 'agentmaddi' or 'user:xxxx'
    toks = []
    for w in q.replace("?"," ").replace("!"," ").replace(","," ").split():
        w = w.strip().lower()
        if not w: continue
        if "agent" in w or "maddi" in w or "user=" in w or "login=" in w:
            toks.append(w)
    return toks

def build_query_string(raw_q: str, explicit_user: str|None):
    # Make the query embedding include identity hint so it matches identity-aware corpus
    hint = f"user={explicit_user} :: " if explicit_user else ""
    return hint + f"question={raw_q}"

def stability_score(similarities: np.ndarray) -> float:
    # similarities: shape [num_paths]
    mu = float(similarities.mean())
    sd = float(similarities.std())
    return mu - STABILITY_STD_WEIGHT * sd

def choose_best_path(path_texts: list[str], embedder, question_vec: np.ndarray):
    # compute pairwise sims to build a stability metric around the question
    # Here: reference is question vector; measure each candidate's sim to it
    cand_vecs = embedder.encode(path_texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = cand_vecs @ question_vec.T  # cosine (normalized)
    score = stability_score(sims.squeeze())
    best_idx = int(np.argmax(sims))
    return best_idx, float(score)

def call_llm(llm, system_prompt: str, user_prompt: str, temperature: float):
    out = llm(
        prompt=f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
        temperature=temperature,
        max_tokens=256,
        stop=["User:", "Assistant:", "\n\nUser:"],
    )
    txt = out["choices"][0]["text"].strip()
    # guardrail: remove any accidental chain-of-thought markers
    CUTS = ["<think>", "</think>", "Thought Process:", "Reasoning:", "Chain-of-Thought:"]
    for c in CUTS:
        if c.lower() in txt.lower():
            txt = txt.split(c, 1)[0].strip()
    return txt

def clamp01(x): return max(0.0, min(1.0, float(x)))

# =========== Main ===========
def main():
    if not Path(DATA_FILE).exists():
        print(f"{neon('⚠️')} {DATA_FILE} not found. Put it next to this script.")
        sys.exit(1)

    print("📦 Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    text_col = pick_column(df, TEXT_COLS)
    user_col = pick_column(df, USER_COLS)
    if not text_col:
        print("❌ Could not find a text column (tried: body_full, body).")
        sys.exit(1)
    if not user_col:
        print("❌ Could not find a user column (tried: login, user_id).")
        sys.exit(1)

    print(f"🧾 Using text column: {text_col} | user column: {user_col}")
    corpus, embs, index, meta = build_or_load_index(df, text_col, user_col)

    # Prepare models
    llm = Llama(model_path=MODEL_PATH, n_ctx=CTX, n_gpu_layers=LAYERS, seed=SEED, verbose=False)
    embedder = SentenceTransformer(EMB_MODEL)

    SYSTEM_PROMPT = (
        "You are Nemotron, a factual reasoning model that answers crisply.\n"
        "Use ONLY the provided context snippets if they answer the question.\n"
        "If unknown, say you don't know from the dataset. Do NOT reveal reasoning steps."
    )

    print(neon("\n🤖 Nemotron + Dataset Probability Stasis v5 (Identity-Aware, #05D9FF)"))
    print("Type 'quit' to exit.\n")

    while True:
        try:
            q = input(neon("💬 You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q: 
            continue
        if q.lower() in {"quit","exit","q"}:
            break

        # Optional: detect explicit user mention (like 'agentmaddi')
        explicit = None
        for col in USER_COLS:
            # If user typed an exact login=value pattern, capture it
            if f"{col}=" in q.lower():
                try:
                    explicit = q.split("=",1)[1].split()[0].strip()
                except:
                    pass
        # Heuristic: if a word looks like a handle, use it
        if explicit is None:
            # scan words; if any exactly match a login in dataset, use it
            words = {w.strip().lower() for w in q.replace("?"," ").replace("!"," ").split()}
            sample_logins = set(str(x).lower() for x in df[user_col].dropna().astype(str).unique())
            hits = words.intersection(sample_logins)
            if hits:
                explicit = sorted(hits, key=len, reverse=True)[0]

        query_text = build_query_string(q, explicit)
        q_vec = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        # Search FAISS
        D, I = index.search(q_vec, max(TOP_K, 15))
        idxs = I[0].tolist()
        dsims = D[0].tolist()

        # Build context block (top-k)
        context_lines = []
        for i, score in zip(idxs, dsims):
            if i < 0: 
                continue
            context_lines.append(f"[sim={score:.3f}] {corpus[i]}")
        context_block = "\n".join(context_lines[:TOP_K])

        # Multi-path LLM sampling on the SAME context
        path_outputs = []
        for t in CANDIDATE_PATHS:
            prompt = (
                f"Context snippets (most similar first):\n{context_block}\n\n"
                f"Question: {q}\n"
                f"Answer in 1-3 sentences. If unknown from snippets, say you don't know."
            )
            path_outputs.append(call_llm(llm, SYSTEM_PROMPT, prompt, t))

        # Pick best via stability (question-anchored)
        best_idx, stab = choose_best_path(path_outputs, embedder, q_vec)
        best = path_outputs[best_idx]

        # UI
        s_norm = clamp01((stab + 1.0) / 2.0)  # put in 0..1 visually
        print(f"\n🏆 Stability [{neon(f'{stab:.3f}')}]", bar(s_norm), badge(s_norm))
        print("\n╭" + "─"*46 + "╮")
        print("Nemotron says:")
        print(best.strip())
        print("╰" + "─"*46 + "╯")
        print("-"*60)

if __name__ == "__main__":
    main()
