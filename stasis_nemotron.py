#!/usr/bin/env python3
"""
Stasis-Optimized Nemotron Inference
Uses your V3.1 filter to select the most stable output from multiple candidates.
"""

import os
import sys
import numpy as np
import pandas as pd
from probv3_1 import ProbabilityStasisV3_1
from llama_cpp import Llama

# === CONFIG ===
MODEL_PATH = "Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"
PROMPT = "Explain the difference between quantum superposition and entanglement in one sentence."
NUM_CANDIDATES = 5
ALPHA_TV = 0.5
BETA_TRANS = 0.1

# === INIT ===
print("🧠 Loading Nemotron model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    verbose=False
)
stasis_engine = ProbabilityStasisV3_1(alpha_tv=ALPHA_TV, beta_trans=BETA_TRANS)

def get_confidence_path(tokens_with_logprobs):
    """Convert logprobs to probabilities (confidence)."""
    logprobs = [t["logprob"] for t in tokens_with_logprobs]
    # Clip to avoid inf/nan
    logprobs = np.clip(logprobs, -20, 0)
    probs = np.exp(logprobs)
    return probs

print(f"\n🎯 Prompt: {PROMPT}\n")
print("🔄 Generating candidates and scoring with Stasis V3.1...\n")

results = []

for i in range(NUM_CANDIDATES):
    # Generate one candidate with logprobs
    output = llm(
        PROMPT,
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        logprobs=1,  # CRITICAL: enables token logprobs
        echo=False
    )
    
    # Extract tokens + logprobs
    tokens_data = output["choices"][0]["logprobs"]["tokens"]
    logprobs = output["choices"][0]["logprobs"]["token_logprobs"]
    
    # Build list of {logprob, token}
    token_logprobs = [{"token": t, "logprob": lp} for t, lp in zip(tokens_data, logprobs) if lp is not None]
    
    if len(token_logprobs) < 2:
        continue  # skip empty/short outputs

    # Get confidence path
    probs = get_confidence_path(token_logprobs)
    
    # Reconstruct text
    text = "".join([t["token"] for t in token_logprobs]).strip()
    
    # Score with Stasis
    score = stasis_engine.stasis_score(probs)
    
    results.append({
        "Candidate": i + 1,
        "Stasis_Score": float(score),
        "Path": probs.tolist(),
        "Message": text
    })
    print(f"✅ Candidate {i+1}: score={score:.4f} | '{text[:60]}...'")

# Select winner
if not results:
    print("❌ No valid candidates generated.")
    sys.exit(1)

df = pd.DataFrame(results)
winner = df.loc[df["Stasis_Score"].idxmax()]

print("\n" + "="*80)
print("🏆 STASIS CHAMPION SELECTED")
print("="*80)
print(f"WINNER: Candidate {winner['Candidate']}")
print(f"STASIS SCORE: {winner['Stasis_Score']:.4f}")
print(f"PATH: {[round(p,3) for p in winner['Path']]}")
print(f"\nFINAL OUTPUT:\n{winner['Message']}")
print("="*80)
