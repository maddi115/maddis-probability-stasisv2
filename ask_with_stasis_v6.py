#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ask_with_stasis_v6.py — Nemotron Probability Stasis v6 (Neon + Parallel + Entropy)
Features:
  • Structured reasoning system prompt
  • 5-path sampling with entropy-style selection (mean(sim) - std(sim))
  • Parallel path generation via ThreadPoolExecutor
  • Embedding cache to speed repeated/near-duplicate queries
  • Neon stability bar + badge + reasoning-type tint
  • GPU pulse (NVML) + latency print
  • CSV logging + session average stability
  • --demo mode to benchmark speeds & stability

Run:
  python3 ask_with_stasis_v6.py                # interactive
  python3 ask_with_stasis_v6.py --demo         # quick self-test/benchmark
"""

import os, sys, time, csv, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Dependencies (graceful fallbacks) ---
try:
    from colorama import Fore, Style, init as color_init
    color_init(autoreset=True)
except Exception:
    class _Dummy: RESET_ALL=""; CYAN=""; BLUE=""; MAGENTA=""
    Fore = Style = _Dummy()
    def color_init(*a, **k): pass

try:
    import psutil  # only for CPU fallback if no NVML
except Exception:
    psutil = None

# NVML GPU usage (optional)
_nvml_ok = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False

import numpy as np

# Sentence-Transformers + llama.cpp
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama

# ------------------ Config ------------------
MODEL_PATH = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
LOG_FILE   = Path("stasis_log.csv")

SYSTEM_PROMPT = """You are Nemotron, a factual reasoning model.
When asked a question:
1) Identify the core phenomenon precisely.
2) Provide a short, logically ordered explanation (2–4 sentences).
3) End with one concise, verified final answer line prefixed with: Final:"""

# Temperatures to probe (wider but still controlled)
TEMPS = [0.60, 0.70, 0.80, 0.85, 0.90]
MAX_TOKENS = 256

# ----------------- Visuals ------------------
def _neon(val: float, low_thr=0.5, high_thr=0.8) -> str:
    # low: cyan, mid: blue, high: magenta
    if val < low_thr:   color = "\033[96m"  # light cyan
    elif val < high_thr: color = "\033[94m"  # electric blue
    else:               color = "\033[95m"  # violet/magenta
    return f"{color}{val:.3f}{Style.RESET_ALL}"

def _badge(score: float) -> str:
    if score >= 0.80: return "🧩 STABLE"
    if score >= 0.50: return "⚖️ MIXED"
    return "🌪️ CHAOTIC"

def _reason_tint(text: str) -> str:
    speculative_tokens = ("might", "could", "may", "possibly", "suggests", "unclear", "unknown")
    spec = any(tok in text.lower() for tok in speculative_tokens)
    return ("\033[95m" if spec else "\033[96m")  # violet speculative, cyan factual

def print_stability(score: float):
    bar_len = max(0, min(20, int(round(score * 20))))
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"\n🏆 Stability [{_neon(score)}] {bar} {_badge(score)}\n")

def gpu_pulse():
    # Returns a string showing current GPU (or CPU) utilization
    try:
        if _nvml_ok:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilp = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            return f"GPU: {utilp}% | VRAM: {used_gb:.1f}/{total_gb:.1f} GB"
        elif psutil:
            return f"CPU: {psutil.cpu_percent(interval=None)}%"
    except Exception:
        pass
    return "GPU: n/a"

# ------------- Models + Caches --------------
print("🤖  Nemotron + Probability Stasis v6 (Neon/Parallel)")
print("Type 'quit' to exit.\n")

llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=40, verbose=False)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# simple in-memory embedding cache
_embed_cache = {}

def embed_text(text: str):
    key = ("emb", text)
    if key in _embed_cache:
        return _embed_cache[key]
    vec = embedder.encode([text], normalize_embeddings=True)[0]
    _embed_cache[key] = vec
    return vec

# -------------- Core Functions --------------
def format_prompt(user_prompt: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"

def run_path(user_prompt: str, temp: float) -> dict:
    start = time.time()
    prompt = format_prompt(user_prompt)
    out = llm(prompt, temperature=temp, max_tokens=MAX_TOKENS)
    txt = out["choices"][0]["text"].strip()
    latency = time.time() - start
    return {"temp": temp, "text": txt, "latency": latency}

def generate_paths_parallel(user_prompt: str, temps=TEMPS):
    results = []
    t0 = time.time()
    # little live pulse while threads run
    with ThreadPoolExecutor(max_workers=len(temps)) as ex:
        futs = {ex.submit(run_path, user_prompt, t): t for t in temps}
        while futs:
            done = {f for f in futs if f.done()}
            for f in done:
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append({"temp": futs[f], "text": f"[ERROR] {e}", "latency": 0.0})
                del futs[f]
            # pulse
            sys.stdout.write("\r⚙️  Running paths... " + gpu_pulse())
            sys.stdout.flush()
            time.sleep(0.08)
    sys.stdout.write("\r" + " " * 60 + "\r")
    total_latency = time.time() - t0
    return results, total_latency

def stability_select(outputs: list) -> dict:
    """
    Compute pairwise cosine sims of answer embeddings.
    Stability score = mean(sim) - std(sim) over all pairwise comparisons.
    Select the single answer whose mean-sim-to-others is highest (ties → lower temp).
    """
    texts = [o["text"] for o in outputs]
    embs = np.stack([embed_text(t) for t in texts], axis=0)
    sims = util.cos_sim(embs, embs).cpu().numpy()

    # For overall session stability: use off-diagonal sims
    mask = ~np.eye(len(outputs), dtype=bool)
    pairwise = sims[mask]
    stability = float(pairwise.mean() - pairwise.std()) if pairwise.size else 0.0

    # Choose the most central answer (highest mean similarity to others)
    mean_sims = sims.sum(axis=1) - 1.0  # subtract self-sim=1
    winner_idx = int(np.argmax(mean_sims))
    winner = outputs[winner_idx].copy()
    winner["stability"] = stability
    winner["winner_index"] = winner_idx
    winner["all_sims_mean"] = float(pairwise.mean()) if pairwise.size else 0.0
    winner["all_sims_std"]  = float(pairwise.std()) if pairwise.size else 0.0
    return winner

def save_log(prompt: str, winner: dict, temps: list, total_latency: float):
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["Timestamp","Prompt","Stability","Temps","WinnerTemp","LatencySec"])
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            prompt,
            f"{winner.get('stability',0.0):.6f}",
            json.dumps(temps),
            f"{winner.get('temp','')}",
            f"{total_latency:.3f}",
        ])

def print_answer_block(winner: dict):
    text = winner["text"]
    shade = _reason_tint(text)
    print("╭──────────────────────────────────────────────╮")
    print("Nemotron says:")
    print(shade + text + Style.RESET_ALL)
    print("╰──────────────────────────────────────────────╯")

def session_avg():
    try:
        # light dependency-free read for mean; uses csv only
        if not LOG_FILE.exists():
            return None
        vals = []
        with LOG_FILE.open("r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                try:
                    vals.append(float(row["Stability"]))
                except Exception:
                    pass
        return (sum(vals)/len(vals)) if vals else None
    except Exception:
        return None

# -------------- CLI / Demo -------------------
def interactive_loop():
    print("llama_context: n_ctx_per_seq (4096) < n_ctx_train (131072) -- the full capacity of the model may not be utilized")
    while True:
        try:
            user = input("💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break
        if not user:
            continue
        if user.lower() in {"quit","exit","q"}:
            print("👋 Exiting.")
            break

        outputs, total_latency = generate_paths_parallel(user, TEMPS)
        winner = stability_select(outputs)

        print_stability(winner["stability"])
        print_answer_block(winner)
        print(f"⏱️ {total_latency:.1f}s | {gpu_pulse()}")
        save_log(user, winner, TEMPS, total_latency)

        avg = session_avg()
        if avg is not None:
            print(f"📈 Average session stability: {avg:.3f}")
        print("-" * 60)

def demo_mode():
    tests = [
        "what is the capital of france",
        "why do cats purr?",
        "what causes auroras?"
    ]
    print("🔬 Running demo (parallel 5-path, entropy-weighted selection)...\n")
    totals = []
    stabs  = []
    for q in tests:
        print(f"💬 {q}")
        outs, tlat = generate_paths_parallel(q, TEMPS)
        win = stability_select(outs)
        print_stability(win["stability"])
        print_answer_block(win)
        print(f"⏱️ {tlat:.1f}s | Winner temp={win['temp']} | {gpu_pulse()}\n")
        totals.append(tlat); stabs.append(win["stability"])
    if totals:
        print("—— Demo Summary ——")
        print(f"Avg Latency: {sum(totals)/len(totals):.2f}s over {len(totals)} prompts")
        print(f"Avg Stability: {sum(stabs)/len(stabs):.3f}")
        print("\nTip: Compare this to your previous v5 runs; v6 should be faster (parallel) and typically higher stability due to structured prompting + entropy selection.")
    print("-" * 60)

# -------------- Entry ------------------------
if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo_mode()
    else:
        interactive_loop()
