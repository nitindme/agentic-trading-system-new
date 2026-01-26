#!/bin/bash
# Launch script for Agentic Trading System Streamlit Dashboard

echo "🚀 Launching Agentic Trading System Dashboard..."
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit plotly
fi

echo ""
echo "🎨 Starting Streamlit dashboard..."
echo "📊 Open your browser to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run streamlit_app.py
