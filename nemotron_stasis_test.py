#!/usr/bin/env python3
"""
nemotron_stasis_test.py — Optimized Nemotron + ProbabilityStasis runner
Features:
 - Progress + ETA
 - CSV logging
 - --limit option for partial runs
 - GPU-optimized (if Nemotron compiled with CUDA)
 - Shorter outputs for faster completion
"""

import csv, random, subprocess, sys, numpy as np, time, argparse
from pathlib import Path
from stasis_core import ProbabilityStasis

# === Config ===
CHAT_CSV = Path("chat_messages.csv")
MODEL_PATH = Path("Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf")
RUN_SCRIPT = Path("run_nemotron.py")
NUM_PATHS = 1        # only one generation path for speed
MAX_TOKENS = "64"    # shorter generations = faster
LAMBDA = 2.0
RESULTS_FILE = Path("nemotron_results.csv")

# === Helpers ===
def run_nemotron_once(prompt: str, temperature: float = 0.8) -> str:
    """Call Nemotron's own runner script for one generation with shorter token limit."""
    cmd = [
        "python3", str(RUN_SCRIPT),
        "--prompt", prompt,
        "--temperature", str(temperature),
        "--n-predict", MAX_TOKENS,
        "--n-gpu-layers", "40"  # use GPU layers if available
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = result.stdout.strip()
        return out.split("Nemotron:", 1)[-1].strip() if "Nemotron:" in out else out
    except Exception as e:
        return f"[error running Nemotron: {e}]"

def simulate_probabilities() -> list[float]:
    """Placeholder for token probs until llama.cpp exposes them."""
    return [random.uniform(0.75, 0.95) for _ in range(8)]

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chat rows to process")
    args = parser.parse_args()

    if not CHAT_CSV.exists():
        print(f"❌ CSV not found: {CHAT_CSV}")
        sys.exit(1)

    lines = [row[0] for row in csv.reader(open(CHAT_CSV, encoding="utf-8")) if row]
    if args.limit:
        lines = lines[:args.limit]
    total = len(lines)

    print(f"🧠 Nemotron GGUF: {MODEL_PATH.name}")
    print(f"📄 Chat file: {CHAT_CSV.name}")
    print(f"💾 Logging to: {RESULTS_FILE}")
    print(f"Total prompts: {total}")
    print(f"⚙️ Using {NUM_PATHS} path(s), max {MAX_TOKENS} tokens per generation")

    stasis = ProbabilityStasis(lambda_instability=LAMBDA, max_keep=1)
    start_all = time.time()

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        writer.writerow(["Prompt", "Best_Path", "Score", "Response"])

        for idx, query in enumerate(lines, 1):
            t0 = time.time()
            print(f"\n[{idx}/{total}] 💬 {query}")
            generations = []
            for i in range(NUM_PATHS):
                temperature = 0.7 + i * 0.1
                print(f"   → generating path {i+1} (temp={temperature}) ...", flush=True)
                gen_text = run_nemotron_once(query, temperature)
                probs = simulate_probabilities()
                generations.append((f"Path {i+1}", probs, [gen_text]))

            filtered = stasis.filter_paths([(n, p) for n, p, _ in generations])
            winner, score = filtered[0]
            best_text = next(t for n, _, t in generations if n == winner)[0]
            elapsed = time.time() - t0
            total_elapsed = time.time() - start_all
            eta = (total_elapsed / idx) * (total - idx)
            print(f"🏆 Best Path: {winner} | Score={score:.3f}")
            print(best_text)
            print(f"⏱️ Row time: {elapsed:.1f}s | ETA ≈ {eta/60:.1f} min remaining")

            writer.writerow([query, winner, f"{score:.3f}", best_text])
            outf.flush()

    total_time = time.time() - start_all
    print(f"\n✅ Done! {total} prompts processed in {total_time/60:.1f} min.")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
