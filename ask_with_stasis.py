#!/usr/bin/env python3
"""
ask_with_stasis.py — Probability Stasis v5 (Full Neon Console)
Nemotron Q&A with semantic stability, breathing pulse, GPU/time overlay, and CSV logging
"""

import os, sys, time, csv, psutil, numpy as np
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util
from colorama import Fore, Style, init
init(autoreset=True)

MODEL_PATH = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
SYSTEM_PROMPT = "You are a helpful, factual AI assistant. Answer clearly and truthfully."
LOG_FILE = "stasis_log.csv"

# --- Neon helper colors ---
def color_stability(score: float) -> str:
    if score < 0.4:
        color = "\033[96m"   # light cyan
    elif score < 0.7:
        color = "\033[94m"   # bright blue
    else:
        color = "\033[95m"   # violet-blue
    return f"{color}{score:.3f}{Style.RESET_ALL}"

def breathing_pulse(text="⚙️  Computing probability stasis"):
    frames = ["▰▱▱▱▱","▰▰▱▱▱","▰▰▰▱▱","▰▰▰▰▱","▰▰▰▰▰","▰▰▰▰▱","▰▰▰▱▱","▰▰▱▱▱"]
    for i in range(8):
        sys.stdout.write(f"\r{Fore.CYAN}{text} {frames[i]}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.12)
    sys.stdout.write("\r" + " " * 60 + "\r")

def print_stability(score: float):
    bar_len = int(score * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"\n🏆 Stability [{color_stability(score)}] {bar}\n")

def get_gpu_usage():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"GPU: {util.gpu}% | VRAM: {mem.used//1e6:.0f}MB"
    except Exception:
        return "GPU: n/a"

# --- Log setup ---
def log_result(prompt, response, score):
    write_header = not Path(LOG_FILE).exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Prompt", "Response", "Stability"])
        writer.writerow([prompt, response.replace("\n"," "), f"{score:.3f}"])

# --- Load models ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=40, verbose=False)

print(f"{Fore.CYAN}🤖  Nemotron + Probability Stasis v5 (Full Neon Console){Style.RESET_ALL}")
print("Type 'quit' to exit.\n")

# --- Main loop ---
while True:
    prompt = input(f"{Fore.CYAN}💬 You:{Style.RESET_ALL} ").strip()
    if prompt.lower() in ["quit", "exit"]:
        print("👋 Exiting.\n")
        break
    if not prompt:
        continue

    breathing_pulse()
    temps = [0.7, 0.8, 0.9]
    outputs = []
    start = time.time()

    for t in temps:
        out = llm(f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:", temperature=t, max_tokens=128)
        text = out["choices"][0]["text"].strip()
        outputs.append(text)

    # semantic similarity
    embs = embedder.encode(outputs, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(embs, embs)
    score = (sims.mean() - np.eye(len(temps)).mean()).item()
    end = time.time()

    print_stability(score)
    best = outputs[int(np.argmax([sims[i].sum().item() for i in range(len(temps))]))]
    gpu_info = get_gpu_usage()
    perf_info = f"⏱️ {end-start:.1f}s | {gpu_info}"

    print(f"\n╭──────────────────────────────────────────────╮")
    print(f"{Fore.MAGENTA}Nemotron says:{Style.RESET_ALL}\n{best[:500]}")
    print(f"╰──────────────────────────────────────────────╯")
    print(f"{Fore.CYAN}{perf_info}{Style.RESET_ALL}")
    print("------------------------------------------------------------\n")

    log_result(prompt, best, score)

