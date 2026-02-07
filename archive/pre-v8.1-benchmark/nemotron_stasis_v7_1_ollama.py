#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import ollama

# Configuration
DATASET = "../data/chat_messages.csv"
MODEL_NAME = "deepseek-r1:8b"  # Using your DeepSeek model
K_NEIGHBORS = 20

# Load dataset
print("📦 Loading dataset...")
df = pd.read_csv(DATASET, encoding="utf-8", on_bad_lines="skip")
text_col = "body_full"
user_col = "login"
print(f"🧾 Using text column: {text_col} | user column: {user_col}")

# Encode dataset
print("🔢 Encoding dataset...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

vectors = encoder.encode(df[text_col].tolist(), show_progress_bar=True, convert_to_numpy=True)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Build FAISS index
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
print(f"✅ FAISS index ready ({len(df)} vectors).")

# Query loop
print(f"\n🤖 Nemotron + Dataset Probability Stasis v7.1 (Ollama)")
print("Type 'quit' to exit.\n")

while True:
    query = input("💬 You: ")
    if query.lower() == "quit":
        break
    
    # Embed query
    q_vec = encoder.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    
    # Search
    sims, ids = index.search(q_vec, K_NEIGHBORS)
    sims, ids = sims[0], ids[0]
    
    # Calculate stability
    weights = np.exp(sims) / np.exp(sims).sum()
    centroid = (weights[:, None] * vectors[ids]).sum(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    stability = (weights * (vectors[ids] @ centroid)).sum()
    confidence = (weights * sims).sum()
    
    # Get evidence
    evidence = df.iloc[ids][text_col].head(5).tolist()
    
    print(f"\n🏆 Stability [{stability:.3f}]")
    print(f"📊 Confidence [{confidence:.3f}]")
    print(f"\n🔍 Top evidence:")
    for i, ev in enumerate(evidence, 1):
        print(f"#{i} {ev[:100]}")
    
    # Generate with Ollama
    if stability > 0.65 and confidence > 0.45:
        prompt = f"""Based on this evidence from chat history, answer the question concisely in 1-2 sentences.
Question: {query}
Evidence:
{chr(10).join(f"- {e}" for e in evidence)}

Answer:"""
        
        print("\n🤖 Nemotron says:")
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        print(response['response'])
    else:
        print("\n⚠️ Insufficient stable evidence for synthesis.")
    
    print("-" * 60)
