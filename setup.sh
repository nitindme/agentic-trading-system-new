#!/bin/bash

# Agentic Trading System - Quick Setup Script
# This script sets up the development environment

set -e  # Exit on error

echo "🤖 Agentic Trading System - Setup"
echo "================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
echo "   This may take 5-10 minutes (downloading PyTorch, transformers, etc.)"
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating project directories..."
directories=(
    "data/raw"
    "data/processed"
    "models/weights"
    "logs"
    "vectorstore"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    touch "$dir/.gitkeep"
done
echo "✅ Directories created"
echo ""

# Copy environment file
echo "🔐 Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created from .env.example"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - HF_TOKEN (Hugging Face)"
    echo "   - ALPACA_API_KEY (for paper trading)"
    echo "   - NEWS_API_KEY (optional)"
else
    echo "✅ .env file already exists"
fi
echo ""

# Download FinBERT model (optional - will download on first use)
echo "🧠 Pre-downloading FinBERT model (optional)..."
read -p "Download FinBERT now? This will speed up first run. (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('Downloading FinBERT...')
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
print('✅ FinBERT downloaded')
    "
fi
echo ""

# Initialize git (if not already)
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Agentic Trading System"
    echo "✅ Git repository initialized"
    echo ""
fi

# Print next steps
echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env and add your API keys"
echo "   2. Test sentiment agent: python agents/sentiment_agent.py"
echo "   3. Test market data: python data/market_data.py"
echo "   4. Run Streamlit UI: streamlit run frontend/app.py"
echo ""
echo "📚 Documentation:"
echo "   - README.md - Full project overview"
echo "   - .env.example - Required API keys"
echo ""
echo "🚀 To activate the environment later, run:"
echo "   source venv/bin/activate"
echo ""
echo "Happy trading! 📈"
