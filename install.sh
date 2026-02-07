#!/bin/bash
# Probability Stasis Installation Script

echo "🔥 Installing Probability Stasis v8.1..."

# Create virtual environment
python3 -m venv stasis-env

# Activate it
source stasis-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "To activate: source stasis-env/bin/activate"
echo "To run: cd core && python3 probability_stasis_v8_1.py"
