#!/usr/bin/env python3
"""
run_nemotron.py — Non-interactive one-shot Nemotron runner
Works with nemotron_stasis_test.py
"""

import argparse, sys, time
from pathlib import Path

# Optional: llama_cpp is used for GGUF inference
try:
    from llama_cpp import Llama
except ImportError:
    print("❌ Missing llama_cpp: install it with 'pip install llama-cpp-python'")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Nemotron single-prompt runner")
    parser.add_argument("--prompt", required=True, help="Prompt text to generate a reply for")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--n-gpu-layers", type=int, default=40)
    args = parser.parse_args()

    model_path = Path("Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    print(f"🧠 Loading model from {model_path} ...", flush=True)
    t0 = time.time()

    # Initialize llama.cpp with GPU offload
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=4096,
        n_threads=8,
        verbose=False,
    )
    print(f"✓ Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Generate one response
    output = llm(
        args.prompt,
        max_tokens=args.n_predict,
        temperature=args.temperature,
        stop=["</s>", "User:", "💬"],
    )

    text = output["choices"][0]["text"].strip()
    print(f"Nemotron: {text}")

if __name__ == "__main__":
    main()
