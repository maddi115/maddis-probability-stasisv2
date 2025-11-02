#!/usr/bin/env python3
# Nemotron Probability Stasis v6_fixed2 — no chain-of-thought leakage, clean context, stability scoring

import time, csv, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from colorama import Style, init
init(autoreset=True)

MODEL_PATH = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
LOG_FILE   = Path("stasis_log.csv")
TEMPS      = [0.55, 0.65, 0.75, 0.82, 0.9]
SEEDS      = [101, 202, 303, 404, 505]
MAX_TOKENS = 220

SYSTEM_PROMPT = """You are Nemotron, a factual reasoning model.
Follow these rules strictly:
- Think briefly, but DO NOT include your analysis or inner thoughts in the output.
- Output ONLY the final answer in clear, concise plain English.
- Do not prefix with 'User:' or 'Assistant:'.
- Do not include <think>...</think> or similar markers."""

STOP_TOKENS = ["</s>", "User:", "\nUser:", "\n\nUser:", "<think>", "</think>"]

print("🤖  Nemotron + Probability Stasis v6_fixed2 (Clean Output)")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
_embed_cache = {}

def embed(text: str):
    if text in _embed_cache: return _embed_cache[text]
    v = embedder.encode([text], normalize_embeddings=True)[0]
    _embed_cache[text] = v
    return v

def color_stab(s):
    if s < 0.5: c = "\033[96m"  # low
    elif s < 0.8: c = "\033[94m"  # medium
    else: c = "\033[95m"  # high
    return f"{c}{s:.3f}{Style.RESET_ALL}"

def badge(s):
    return "🧩 STABLE" if s >= 0.8 else ("⚖️ MIXED" if s >= 0.5 else "🌪️ CHAOTIC")

def clean_text(txt: str) -> str:
    # strip XML-ish think tags
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL|re.IGNORECASE)
    # remove role markers
    txt = re.sub(r"^(User:|Assistant:)\s*", "", txt, flags=re.IGNORECASE|re.MULTILINE)
    # collapse excess whitespace
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt

def run_paths(prompt: str):
    outs = []
    for t, seed in zip(TEMPS, SEEDS):
        # fresh context per path avoids contamination
        llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=40, verbose=False)
        out = llm(
            f"{SYSTEM_PROMPT}\n\nQuestion: {prompt}\nAnswer:",
            temperature=t,
            max_tokens=MAX_TOKENS,
            stop=STOP_TOKENS,
            seed=seed,
            repeat_penalty=1.05,
            frequency_penalty=0.2,
            presence_penalty=0.0,
        )
        raw = out["choices"][0]["text"]
        txt = clean_text(raw)
        outs.append({"t": t, "seed": seed, "txt": txt})
        del llm
    return outs

def choose_best(outs):
    embs = np.stack([embed(o["txt"]) for o in outs])
    sims = util.cos_sim(embs, embs).cpu().numpy()
    mask = ~np.eye(len(outs), dtype=bool)
    stability = float(sims[mask].mean() - sims[mask].std())
    # choose output most similar to others
    mean_s = sims.sum(1) - 1.0
    win = int(np.argmax(mean_s))
    return outs[win]["txt"], outs[win]["t"], outs[win]["seed"], stability

def loop():
    while True:
        q = input("💬 You: ").strip()
        if q.lower() in {"quit","exit"}: break
        t0 = time.time()
        outs = run_paths(q)
        final, temp, seed, stab = choose_best(outs)
        dur = time.time() - t0

        bar_len = max(0, min(20, int(stab * 20)))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"\n🏆 Stability [{color_stab(stab)}] {bar} {badge(stab)}")
        print("╭──────────────────────────────────────────────╮")
        print("Nemotron says:\n" + final)
        print("╰──────────────────────────────────────────────╯")
        print(f"⏱️ {dur:.1f}s | Temp {temp} | Seed {seed}")
        print("------------------------------------------------------------")

        try:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([time.strftime("%F %T"), q, f"{stab:.4f}", temp, seed, f"{dur:.2f}"])
        except Exception:
            pass

if __name__ == "__main__":
    loop()
