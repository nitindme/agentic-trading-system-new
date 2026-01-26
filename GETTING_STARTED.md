# 🚀 Getting Started with Agentic Trading System

This guide will help you set up and run the agentic trading system in **15 minutes**.

---

## 📋 Prerequisites

### Required
- **Python 3.10+** (check: `python3 --version`)
- **pip** (Python package manager)
- **git** (version control)

### Optional (for full features)
- **CUDA-capable GPU** (for faster sentiment analysis)
- **Docker** (for containerized deployment)

---

## ⚡ Quick Start (5 Minutes)

### 1. Clone & Setup

```bash
cd /Users/nitindigraje/Documents
git clone https://github.com/nitindme/agentic-trading-system.git
cd agentic-trading-system

# Run automated setup
./setup.sh
```

The setup script will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Copy `.env.example` to `.env`

### 2. Configure API Keys

Edit `.env` file and add your API keys:

```bash
# Open in your editor
code .env  # VS Code
# or
nano .env  # Terminal editor
```

**Minimum required** (for demo):
```env
OPENAI_API_KEY=sk-...              # Get from https://platform.openai.com
HF_TOKEN=hf_...                     # Get from https://huggingface.co/settings/tokens
```

**For full features**:
```env
ALPACA_API_KEY=PK...                # Paper trading - https://alpaca.markets
ALPACA_SECRET_KEY=...
NEWS_API_KEY=...                    # News data - https://newsapi.org
```

### 3. Test Installation

```bash
# Activate environment
source venv/bin/activate

# Test sentiment agent
python agents/sentiment_agent.py

# Test market data
python data/market_data.py
```

If both run without errors, you're ready! 🎉

---

## 🧪 Demo Mode (No API Keys Required)

Want to test without API keys? Use the mock data mode:

```bash
# Set demo mode in .env
EXECUTION_MODE=SIMULATION
USE_MOCK_DATA=true

# Run with sample data
python demo/run_demo.py
```

---

## 📊 Usage Examples

### Example 1: Analyze a Stock

```python
from agents.sentiment_agent import NewsSentimentAgent
from data.market_data import MarketDataProvider

# Initialize agents
sentiment_agent = NewsSentimentAgent()
market_data = MarketDataProvider()

# Analyze Apple
symbol = "AAPL"
news = market_data.get_news(symbol)
sentiment = sentiment_agent.analyze_news_batch(news)

print(f"Sentiment: {sentiment.label} ({sentiment.score:.2f})")
print(f"Confidence: {sentiment.confidence:.2%}")
```

### Example 2: Run Full Analysis (Coming in Phase 2)

```python
from graph.workflow import TradingWorkflow

workflow = TradingWorkflow()
result = workflow.analyze_stock("TSLA", mode="SIMULATION")

print(result)
```

---

## 🎯 Development Phases

### ✅ Phase 1: Foundation (Current)
- [x] Project structure
- [x] Sentiment agents (FinBERT)
- [x] Market data integration
- [ ] Basic Streamlit UI
- [ ] Unit tests

**What you can do now**:
- Analyze news sentiment for any stock
- Fetch real-time market data
- Test individual agents

### 📋 Phase 2: Intelligence (Next)
- [ ] Technical analysis agent
- [ ] Fundamental analysis agent
- [ ] Risk/critic agent
- [ ] LangGraph workflow
- [ ] Decision engine

**What you'll be able to do**:
- Multi-agent analysis
- BUY/HOLD/SELL recommendations
- Confidence scoring

### 📋 Phase 3: Execution
- [ ] Alpaca paper trading
- [ ] Risk guardrails
- [ ] Order management
- [ ] Execution logging

**What you'll be able to do**:
- Simulate trades
- Test strategies
- Track performance

### 📋 Phase 4: Production
- [ ] Complete Streamlit UI
- [ ] FastAPI backend
- [ ] Docker deployment
- [ ] Monitoring & logging

**What you'll be able to do**:
- Full web interface
- Real-time monitoring
- Production deployment

---

## 🛠️ Project Structure

```
agentic-trading-system/
├── agents/                   # AI agents
│   ├── sentiment_agent.py   # ✅ News + social sentiment
│   ├── technical_agent.py   # 📋 Technical analysis
│   ├── fundamental_agent.py # 📋 Fundamental checks
│   └── risk_agent.py        # 📋 Risk validation
├── data/                    # Data providers
│   └── market_data.py       # ✅ Market data fetcher
├── graph/                   # LangGraph workflows
│   └── workflow.py          # 📋 Agent orchestration
├── models/                  # ML models
│   └── sentiment_model.py   # 📋 Model wrappers
├── execution/               # Trade execution
│   └── broker_api.py        # 📋 Alpaca integration
├── frontend/                # UI
│   └── app.py               # 📋 Streamlit interface
└── tests/                   # Unit tests
    └── test_agents.py       # 📋 Agent tests
```

**Legend**: ✅ Complete | 🔄 In Progress | 📋 Planned

---

## 🐛 Troubleshooting

### Issue: Import errors for torch/transformers

**Solution**: Make sure you activated the virtual environment
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: CUDA out of memory

**Solution**: Use CPU instead (slower but works)
```python
# In sentiment_agent.py
agent = NewsSentimentAgent(device="cpu")
```

### Issue: API rate limits

**Solution**: Add delays between requests
```env
# In .env
API_RATE_LIMIT_DELAY=1  # 1 second between calls
```

### Issue: News data not found

**Solution**: Check API key and symbol
```python
# Verify API key is set
import os
print(os.getenv("NEWS_API_KEY"))

# Try popular symbol
news = provider.get_news("AAPL")
```

---

## 📚 Key Concepts

### Sentiment Analysis
- **FinBERT**: Financial domain-specific BERT model
- **Score Range**: -1 (very negative) to +1 (very positive)
- **Confidence**: 0 to 1 (model certainty)

### Agent Architecture
- **Agents**: Specialized AI components (sentiment, technical, risk)
- **LangGraph**: Orchestrates agent interactions
- **State**: Shared data between agents

### Risk Management
- **Guardrails**: Rules that prevent unsafe trades
- **Confidence Threshold**: Minimum confidence for execution
- **Position Sizing**: Max % of portfolio per trade

---

## 🎓 Learning Resources

### Financial AI
- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [Sentiment Analysis in Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489963)

### LangChain/LangGraph
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)
- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/)

### Trading & Risk
- [Algorithmic Trading Basics](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp)
- [Risk Management](https://www.investopedia.com/terms/r/riskmanagement.asp)

---

## 🤝 Contributing

This is a capstone/portfolio project, but contributions are welcome!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Additional sentiment models
- More technical indicators
- Alternative data sources
- UI improvements
- Documentation

---

## 📞 Support

### Got Questions?
- 📧 Email: [your-email]
- 💬 GitHub Issues: [Report a bug or request feature]
- 🐦 Twitter: [@your-handle]

### Common Questions

**Q: Can I use this for real trading?**
A: The system defaults to paper trading (simulation). Real trading requires explicit opt-in and is at your own risk.

**Q: What stocks can I analyze?**
A: Any stock available on yfinance (US markets, some international)

**Q: Do I need a GPU?**
A: No, but it speeds up sentiment analysis. CPU works fine for demos.

**Q: Is this financial advice?**
A: No, this is educational software for learning AI/ML in finance.

---

## ✅ Next Steps

Now that you're set up:

1. **✅ Run the sentiment analysis demo**
   ```bash
   python agents/sentiment_agent.py
   ```

2. **✅ Test with your favorite stock**
   ```python
   # Edit the symbol in the file
   symbol = "TSLA"  # or any other
   ```

3. **📋 Star the repo** (if you find it useful!)
   ```bash
   # On GitHub
   https://github.com/nitindme/agentic-trading-system
   ```

4. **📋 Join the discussion**
   - Share your results
   - Request features
   - Report bugs

---

**Ready to build?** Let's move to Phase 2: [Implement LangGraph Workflow →](docs/phase2_workflow.md)

Happy trading! 📈🤖
