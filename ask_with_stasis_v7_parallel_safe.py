#!/usr/bin/env python3
# Nemotron + Probability Stasis v7 (Parallel-Safe, Neon, Entropy-Weighted)
# - Process-based parallelism (safe for llama.cpp CUDA allocator)
# - Auto VRAM-aware concurrency (falls back to sequential if tight)
# - Structured reasoning prompt
# - Entropy-weighted stability scoring: mean(sim) - 0.5*std(sim) - small penalties
# - Clean output (no <think>, no "User:" bleed)
# - Neon stability bar + badge

import os, sys, time, re, math, traceback
from pathlib import Path
from statistics import mean, pstdev
from itertools import combinations
from typing import List, Tuple

# --- Optional GPU/VRAM sensing (best-effort) ---
def gpu_info():
    try:
        # prefer nvidia-ml-py (modern); fall back to pynvml if present
        try:
            import nvidia_ml_py as nv
        except Exception:
            import pynvml as nv  # deprecated warning is fine
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        total = getattr(mem, "total", 0)
        used  = getattr(mem, "used", 0)
        free  = getattr(mem, "free", max(total-used,0))
        gpu   = getattr(util, "gpu", 0)
        nv.nvmlShutdown()
        return dict(ok=True, total=total, used=used, free=free, util=gpu)
    except Exception:
        return dict(ok=False, total=0, used=0, free=0, util=0)

# --- Pretty console (neon) ---
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    CYN = "\033[96m"; BLU = "\033[94m"; VIO = "\033[95m"; DIM = "\033[90m"
    RST = Style.RESET_ALL
except Exception:
    CYN = BLU = VIO = DIM = RST = ""

def neon_score(score: float) -> str:
    if score < 0.5: col = CYN
    elif score < 0.8: col = BLU
    else: col = VIO
    return f"{col}{score:.3f}{RST}"

def badge(score: float) -> str:
    if score >= 0.80: return "🧩 STABLE"
    if score >= 0.55: return "⚖️ MIXED"
    return "🌪️ CHAOTIC"

def print_bar(score: float):
    filled = int(max(0,min(1,score))*20)
    bar = "█"*filled + "░"*(20-filled)
    print(f"\n🏆 Stability [{neon_score(score)}] {bar} {badge(score)}\n")

def neon_box(text: str):
    print("╭" + "─"*46 + "╮")
    print("Nemotron says:")
    # Wrap to ~90 cols but keep it simple
    for line in re.sub(r'\s+\n', '\n', text.strip()).splitlines():
        print(line)
    print("╰" + "─"*46 + "╯")

# --- LLM + Embeddings ---
MODEL_PATH = os.environ.get("NEMOTRON_MODEL", "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf")
SYSTEM_PROMPT = """You are Nemotron, a factual reasoning model.
When asked a question:
1) Think briefly BEFORE you answer.
2) Provide clear, concise reasoning in 1–3 sentences.
3) Then give a short final answer.
Only answer the user's latest question.
Avoid meta-chatter and internal notes.
"""

# Temperatures we will sample. Keep compact to reduce VRAM.
TEMPS = [0.62, 0.70, 0.78, 0.84, 0.90]
MAX_PATHS = 5

# Safety defaults for llama.cpp when multiple processes run
LLAMA_KW = dict(
    n_ctx=4096,
    n_gpu_layers=24,     # reduce per-process VRAM
    seed=0,
    use_mmap=True,
    use_mlock=False,
    low_vram=True,
    verbose=False
)

def clean_output(txt: str) -> str:
    # Strip common thinking/role artifacts & unrelated injected turns
    t = txt.strip()
    # Remove XML-ish thinking tags
    t = re.sub(r'<\s*/?\s*(think|thought|reasoning)\s*>', '', t, flags=re.I)
    # Remove "User:" / "Assistant:" blocks that bled in
    t = re.sub(r'(?s)\b(User|Assistant)\s*:\s*.*', '', t)
    # Remove code fences if hallucinated
    t = re.sub(r'```.*?```', '', t, flags=re.S)
    # Remove excess blank lines
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def grade_penalty(txt: str) -> float:
    # Light penalties for known low-quality patterns
    lower = txt.lower()
    pen = 0.0
    if "i'm not sure" in lower or "as an ai" in lower:
        pen += 0.08
    # Overlong ramble penalty (>1000 chars)
    if len(txt) > 1000:
        pen += min(0.10, (len(txt)-1000)/4000.0)
    # Off-topic hints
    if "homework" in lower or "submitter" in lower or "ignore previous" in lower:
        pen += 0.12
    return pen

# Fork-safe worker (one Llama per PROCESS)
def _worker_generate(args) -> Tuple[float, str]:
    t, user_prompt, seed = args
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=MODEL_PATH, **LLAMA_KW)
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"
        out = llm(prompt,
                  temperature=float(t),
                  top_p=0.92,
                  max_tokens=384,
                  stop=["User:", "\n\nUser:", "\nAssistant:"],
                  seed=int(seed))
        # Support dict or str return depending on binding version
        if isinstance(out, dict):
            text = out.get("choices",[{}])[0].get("text","")
        else:
            text = str(out)
        return (t, clean_output(text)[:2000])
    except Exception as e:
        return (t, f"[error at temp {t}: {e}]")

def pairwise_similarities(embs) -> List[float]:
    # cosine similarities for all unique pairs
    import numpy as np
    sims = []
    for i, j in combinations(range(len(embs)), 2):
        a = embs[i] / (np.linalg.norm(embs[i]) + 1e-8)
        b = embs[j] / (np.linalg.norm(embs[j]) + 1e-8)
        sims.append(float((a*b).sum()))
    return sims

def choose_best(candidates: List[str]) -> Tuple[int, float]:
    # Embed all, compute entropy-weighted stability:
    # score_i = mean(sim(i, others)) - 0.5*std(sim(all pairs)) - penalty(i)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer('all-MiniLM-L6-v2')
    vecs = emb.encode(candidates, convert_to_numpy=True, normalize_embeddings=False)

    # all-pair std gauges global inconsistency; we reuse for each candidate
    all_pair_sims = pairwise_similarities(vecs)
    global_std = pstdev(all_pair_sims) if len(all_pair_sims) >= 2 else 0.0

    # per-candidate mean similarity to others
    import numpy as np
    best_idx, best_score = 0, -1e9
    for i in range(len(candidates)):
        sims_i = []
        vi = vecs[i]
        for j in range(len(candidates)):
            if i == j: continue
            vj = vecs[j]
            a = vi/(np.linalg.norm(vi)+1e-8); b = vj/(np.linalg.norm(vj)+1e-8)
            sims_i.append(float((a*b).sum()))
        mu = mean(sims_i) if sims_i else 0.0
        pen = grade_penalty(candidates[i])
        score = mu - 0.5*global_std - pen
        if score > best_score:
            best_idx, best_score = i, score
    # Normalize score from cosine-ish range to 0..1-ish for the bar (rough)
    norm = (best_score + 1.0)/2.0
    return best_idx, max(0.0, min(1.0, norm))

def concurrency_from_vram() -> int:
    info = gpu_info()
    if not info["ok"]:
        return 2  # conservative default on unknown
    # With 4B Q6_K and n_gpu_layers=24, expect ~2.2–2.8 GB per proc.
    # Keep margin; aim for <= 70% usage.
    free_gb = info["free"] / (1024**3)
    if free_gb >= 6.0: return 3
    if free_gb >= 3.5: return 2
    return 1

def run_paths(user_prompt: str) -> Tuple[str, float, List[Tuple[float,str]]]:
    temps = TEMPS[:MAX_PATHS]
    # Decide safe level of parallel
    k = concurrency_from_vram()
    # Build jobs with different seeds for diversity
    jobs = [(t, user_prompt, int(time.time()*1000) % 2_147_483_647 + i*97) for i, t in enumerate(temps)]

    outputs: List[Tuple[float,str]] = []

    if k <= 1:
        # Sequential (safe)
        for a in jobs:
            outputs.append(_worker_generate(a))
    else:
        # Process-based pool; spawn method is safer for CUDA contexts
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=k) as pool:
            for res in pool.imap_unordered(_worker_generate, jobs):
                outputs.append(res)

    # Sort back by temperature order for logging readability
    outputs.sort(key=lambda x: temps.index(x[0]) if x[0] in temps else 999)

    # Extract candidate texts
    candidates = [o[1] for o in outputs]
    # Fallback: if everything error'd, return best message
    if not any(c and not c.startswith("[error") for c in candidates):
        best = max(candidates, key=lambda s: len(s))
        return best, 0.25, outputs

    # Choose best via entropy-weighted stability
    try:
        idx, stable = choose_best(candidates)
        return candidates[idx], stable, outputs
    except Exception:
        # If embedding model not available for any reason, pick middle temp
        mid = len(candidates)//2
        return candidates[mid], 0.50, outputs

def main():
    print("🤖  Nemotron + Probability Stasis v7 (Parallel-Safe)")
    print("Type 'quit' to exit.\n")
    # Gentle notice if model path missing
    if not Path(MODEL_PATH).exists():
        print(f"{DIM}⚠️  Model not found at: {MODEL_PATH}{RST}")
        print("   Set NEMOTRON_MODEL=/path/to/model.gguf or edit MODEL_PATH.\n")

    while True:
        try:
            user = input(f"{Fore.WHITE if Fore else ''}💬 You: {RST}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye")
            break
        if user.lower() in {"quit", "exit"}:
            print("👋 Bye")
            break
        if not user:
            continue

        # Show quick GPU heads-up
        gi = gpu_info()
        if gi["ok"]:
            free_gb = gi["free"]/(1024**3)
            util = gi["util"]
            print(f"{DIM}⚙️  Running paths... GPU: {util}% | VRAM free: {free_gb:.1f} GB{RST}")

        t0 = time.time()
        try:
            answer, stability, raw = run_paths(user)
        except Exception as e:
            print(f"\n{Fore.RED if Fore else ''}Error: {e}{RST}")
            traceback.print_exc()
            continue
        dt = time.time()-t0

        print_bar(stability)
        neon_box(answer)
        print("------------------------------------------------------------")
        # tiny footer
        k = concurrency_from_vram()
        print(f"{DIM}⏱️ {dt:.1f}s | Mode: {'Parallel x'+str(k) if k>1 else 'Sequential'} | Temps: {', '.join(map(str,TEMPS[:MAX_PATHS]))}{RST}\n")

if __name__ == "__main__":
    main()
