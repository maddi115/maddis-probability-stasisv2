# Project Structure
```
maddis-probability-stasisv2/
├── core/
│   ├── nemotron_stasis_v8_1_streaming_autotune.py  ← PRODUCTION SYSTEM
│   └── stasis_core.py                               ← Core utilities
├── data/
│   ├── .gitkeep                                     ← Preserve directory
│   └── chat_messages.csv                            ← Your dataset (gitignored)
├── archive/
│   ├── pre-v8.1-benchmark/
│   │   ├── BENCHMARK_RESULTS.md                    ← Full benchmark analysis
│   │   ├── nemotron_stasis_v7_1.py                 ← LLM version (archived)
│   │   ├── nemotron_stasis_v7_1_ollama.py          ← Ollama version (archived)
│   │   └── nemotron_stasis_v8_fieldmap.py          ← Field map (archived)
│   └── interactive/                                 ← Old interactive versions
├── docs/
│   └── readme.txt                                   ← Technical documentation
├── scripts/
│   └── [various test scripts]
├── .gitignore                                       ← Comprehensive ignore rules
├── requirements.txt                                 ← LLM-free dependencies
├── README.md                                        ← Main documentation
└── PROJECT_STRUCTURE.md                            ← This file
```

## Active Files Only

**Production System**: `core/nemotron_stasis_v8_1_streaming_autotune.py`
**Dependencies**: See `requirements.txt` (LLM-free)
**Documentation**: `docs/readme.txt` + `archive/pre-v8.1-benchmark/BENCHMARK_RESULTS.md`

## Gitignored (won't be committed)

- Virtual environments (`venv/`, `nemotron-env/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Data files (`data/*.csv`)
- FAISS indices (`*.faiss`, `.v8_1_stream_state.json`)
- Logs (`v8_1_runs.csv`)
- Models (removed entirely - no longer needed)

## LLM Removal

All LLM dependencies removed post-benchmark:
- No `llama-cpp-python`
- No `ollama`
- No model files
- Pure semantic field reasoning only

System is now **100% LLM-free** ✨
