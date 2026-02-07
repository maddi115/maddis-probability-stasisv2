# 🔥 Probability Stasis - Dataset-Driven Reasoning Framework

**Production Version**: v8.1_streaming_autotune  
**Status**: Benchmark-proven, zero hallucinations ✅

## What is This?

Probability Stasis replaces LLM generation with semantic field analysis. Instead of guessing answers, it measures coherence in your dataset and only returns verified information.

## Quick Start
```bash
# Setup
python3 -m venv nemotron-env
source nemotron-env/bin/activate
pip install -r requirements.txt

# Run
cd core
python3 nemotron_stasis_v8_1_streaming_autotune.py
```

## Why v8.1 Won

Tested against LLM-based v7 systems:
- ✅ **Zero hallucinations** (perfect precision)
- ✅ **Auto-tuning** improves stability +0.007 to +0.037 per query
- ✅ **Smart abstention** - says "I don't know" instead of lying
- ✅ **10x faster** - no LLM inference required
- ✅ **Deterministic** - same query = same answer

See `archive/pre-v8.1-benchmark/BENCHMARK_RESULTS.md` for full analysis.

## Core Principle

> "A language model should not guess what the data already knows." — maddi

Semantic coherence > Token prediction

## Features

- **Streaming Updates**: Incremental FAISS index (no rebuilds)
- **Auto-Tuned Entropy**: Binary search for optimal temperature
- **Evidence Weighting**: Softmax-weighted semantic centroids  
- **Stability Metrics**: Quantified semantic agreement [0-1]
- **Comprehensive Logging**: All metrics saved to CSV

## Documentation

- `docs/readme.txt` - Full technical documentation
- `archive/pre-v8.1-benchmark/BENCHMARK_RESULTS.md` - Benchmark results
- `core/nemotron_stasis_v8_1_streaming_autotune.py` - Production system

## License

Research prototype - use at your own risk 🔥
