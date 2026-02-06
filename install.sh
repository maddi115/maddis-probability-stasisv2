#!/usr/bin/env bash
set -e
echo "=== 🧠 Maddi’s Probability Stasis v2 Installer ==="

# ---- ENV SETUP ----
echo "[1/5] Creating virtual environment..."
python3 -m venv nemotron-env
source nemotron-env/bin/activate

echo "[2/5] Upgrading pip..."
pip install --upgrade pip wheel setuptools numpy pandas faiss-cpu sentence-transformers colorama

# ---- DEPENDENCIES ----
echo "[3/5] Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 numpy pandas faiss-cpu sentence-transformers colorama
pip install transformers==4.45.0 \ numpy pandas faiss-cpu sentence-transformers colorama
            faiss-cpu \
            sentence-transformers \
            scikit-learn \
            pandas numpy tqdm matplotlib colorama rich umap-learn seaborn

# ---- REPO CLEANUP ----
echo "[4/5] Cleaning old caches..."
rm -rf __pycache__ */__pycache__ .pytest_cache build dist *.egg-info *.log *.tmp || true

# ---- TEST LAUNCH ----
echo "[5/5] Verifying installation..."
python3 -c "import torch, faiss, transformers, pandas, numpy; print('✅ Core libs loaded successfully.')"

echo ""
echo "🎯 Setup complete! To activate:"
echo "   source nemotron-env/bin/activate"
echo ""
echo "Run your model:"
echo "   python3 run_nemotron.py"
echo ""
echo "✨ Maddi’s Probability Stasis v2 environment ready."
