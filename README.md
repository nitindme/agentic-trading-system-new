# 🤖 Agentic Stock Sentiment & Trade Execution System

> **An intelligent AI trading assistant that analyzes market sentiment, validates signals, and executes trades with professional guardrails**

⚠️ **Ethical Framework**: This system produces **decision support signals**, not financial advice. All trades require explicit approval or rule-based validation.

---

## 🎯 Project Overview

### What It Does

1. **Sentiment Analysis**: Analyzes news, social media, and earnings data using domain-specific AI models
2. **Multi-Agent Validation**: Technical analysis, fundamental checks, and risk assessment
3. **Agentic Decision-Making**: LangGraph orchestrates agents with conditional routing
4. **Safe Execution**: Paper trading with configurable guardrails and human-in-the-loop controls

### Why It Matters

- **Real-world complexity**: Production-grade architecture with RAG, agents, and risk management
- **Ethical AI design**: Built-in safety controls and explainability
- **Financial domain expertise**: Uses specialized models (FinBERT) and broker APIs
- **Interview-ready**: Demonstrates advanced LangChain, LangGraph, and system design skills

---

## 🏗️ Architecture

### Agent System (LangGraph)

```
User Input (Stock Symbol)
        ↓
┌───────────────────────────────┐
│  Sentiment Intelligence Layer │
├───────────────────────────────┤
│ • News Sentiment Agent        │
│ • Social Sentiment Agent      │
│ • RAG Knowledge Base          │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│    Validation Layer           │
├───────────────────────────────┤
│ • Technical Analysis Agent    │
│ • Fundamental Check Agent     │
│ • Risk/Critic Agent           │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│    Decision Engine            │
├───────────────────────────────┤
│ • Aggregates all signals      │
│ • Produces recommendation     │
│ • Calculates confidence       │
└───────────────────────────────┘
        ↓
    [Confidence ≥ Threshold?]
        ↓ YES              ↓ NO
┌───────────────┐   ┌─────────────┐
│ Execution     │   │ Reject +    │
│ Agent         │   │ Explain     │
│ (Paper Trade) │   │ Reasoning   │
└───────────────┘   └─────────────┘
```

### Agent Responsibilities

| Agent | Purpose | Output |
|-------|---------|--------|
| **News Sentiment** | Analyzes financial news, earnings reports | Sentiment score + confidence |
| **Social Sentiment** | Twitter/Reddit/StockTwits analysis | Social score + hype detection |
| **Technical Analysis** | RSI, MACD, moving averages, volume | Technical signals |
| **Fundamental Check** | P/E ratio, earnings growth, debt levels | Fundamental health score |
| **Risk/Critic** | Validates all signals, rejects unsafe trades | Go/No-go decision |
| **Decision Engine** | Produces final recommendation | BUY/HOLD/SELL + reasoning |
| **Execution** | Places trade via broker API | Order confirmation |

---

## 🧠 Technology Stack

### AI/ML
- **LLMs**: Hugging Face (FinBERT, Llama 3)
- **Sentiment Models**: 
  - `ProsusAI/finbert` (financial sentiment)
  - `yiyanghkust/finbert-tone` (tone analysis)
  - SentenceTransformers (embeddings)
- **Agent Framework**: LangChain + LangGraph
- **RAG**: LlamaIndex (earnings calls, SEC filings, news)
- **Vector DB**: FAISS / ChromaDB

### Backend
- **API**: FastAPI (async)
- **Market Data**: yfinance, Alpha Vantage
- **Broker Integration**: Alpaca (paper trading)
- **Caching**: Redis (optional)

### Frontend
- **UI**: Streamlit (demo interface)
- **Visualization**: Plotly, matplotlib

### Monitoring
- **Agent Observability**: LangSmith
- **Logging**: Structured logs (JSON)
- **Metrics**: Custom risk metrics

---

## 🛡️ Safety Guardrails

### Risk Controls

```python
GUARDRAILS = {
    "min_confidence": 0.75,          # Reject trades below 75% confidence
    "max_trade_size": 0.05,          # Max 5% of portfolio per trade
    "max_daily_trades": 3,           # Cool-down period
    "volatility_threshold": 0.30,    # Reject if volatility > 30%
    "conflicting_signals": False,    # Require agent alignment
    "human_approval": True           # Default: human-in-the-loop
}
```

### Execution Modes

1. **SIMULATION** (default): Paper trading only, no real money
2. **APPROVAL_REQUIRED**: System proposes, human approves
3. **AUTO_EXECUTE**: Rule-based automation (requires explicit opt-in)

---

## 📋 Project Structure

```
agentic-trading-system/
├── agents/
│   ├── sentiment_agent.py       # News + social sentiment
│   ├── technical_agent.py       # TA indicators
│   ├── fundamental_agent.py     # Fundamental analysis
│   ├── risk_agent.py            # Risk validation
│   ├── decision_agent.py        # Final recommendation
│   └── execution_agent.py       # Trade placement
├── graph/
│   ├── workflow.py              # LangGraph orchestration
│   └── state.py                 # Shared state definitions
├── rag/
│   ├── knowledge_base.py        # LlamaIndex setup
│   ├── embeddings.py            # Vector embeddings
│   └── retrieval.py             # RAG queries
├── models/
│   ├── sentiment_model.py       # FinBERT wrapper
│   └── ensemble.py              # Multi-model ensemble
├── data/
│   ├── market_data.py           # yfinance integration
│   ├── news_scraper.py          # News API
│   └── social_scraper.py        # Twitter/Reddit (optional)
├── execution/
│   ├── broker_api.py            # Alpaca integration
│   ├── order_manager.py         # Order validation
│   └── risk_manager.py          # Guardrail enforcement
├── backend/
│   ├── main.py                  # FastAPI server
│   ├── models.py                # Pydantic schemas
│   └── routes.py                # API endpoints
├── frontend/
│   ├── app.py                   # Streamlit UI
│   └── components/              # UI components
├── prompts/
│   ├── sentiment_prompts.py
│   ├── decision_prompts.py
│   └── explanation_prompts.py
├── utils/
│   ├── logging.py
│   ├── metrics.py
│   └── validators.py
├── tests/
│   ├── test_agents.py
│   ├── test_workflow.py
│   └── test_risk_manager.py
├── notebooks/
│   ├── sentiment_exploration.ipynb
│   └── backtesting.ipynb
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/nitindme/agentic-trading-system.git
cd agentic-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configuration

```bash
# Required API keys (add to .env)
OPENAI_API_KEY=your_key_here          # For LLM reasoning
HF_TOKEN=your_token_here              # Hugging Face models
ALPACA_API_KEY=your_key_here          # Paper trading
ALPACA_SECRET_KEY=your_secret_here
NEWS_API_KEY=your_key_here            # News data
LANGSMITH_API_KEY=your_key_here       # Agent observability (optional)
```

### 3. Run the System

```bash
# Option 1: Streamlit UI (recommended for demo)
streamlit run frontend/app.py

# Option 2: FastAPI backend
uvicorn backend.main:app --reload

# Option 3: Jupyter notebook (experimentation)
jupyter notebook notebooks/sentiment_exploration.ipynb
```

---

## 🎮 Usage Examples

### Example 1: Analyze Stock Sentiment

```python
from graph.workflow import TradingWorkflow

# Initialize workflow
workflow = TradingWorkflow()

# Analyze stock
result = workflow.analyze_stock(
    symbol="AAPL",
    mode="SIMULATION"
)

print(result)
# Output:
# {
#   "symbol": "AAPL",
#   "recommendation": "BUY",
#   "confidence": 0.82,
#   "sentiment": {
#     "news": 0.78,
#     "social": 0.65,
#     "overall": 0.72
#   },
#   "technical": {
#     "rsi": 58.3,
#     "trend": "BULLISH"
#   },
#   "reasoning": [
#     "Positive earnings surprise (+15% YoY)",
#     "Bullish RSI (58.3)",
#     "Strong institutional buying"
#   ],
#   "risk_level": "MEDIUM",
#   "execution_status": "SIMULATED"
# }
```

### Example 2: Execute Trade with Approval

```python
# Run analysis
recommendation = workflow.analyze_stock("TSLA")

# Review recommendation
if recommendation["confidence"] > 0.75:
    # Execute trade (paper trading)
    order = workflow.execute_trade(
        symbol="TSLA",
        action=recommendation["recommendation"],
        quantity=10,
        mode="SIMULATION"
    )
    print(f"Order placed: {order}")
```

---

## 📊 Key Features

### 1. Multi-Source Sentiment Analysis
- Financial news (Bloomberg, Reuters, WSJ)
- Earnings call transcripts
- Social media (Twitter/Reddit)
- SEC filings (10-K, 10-Q)

### 2. Domain-Specific AI Models
- **FinBERT**: Financial sentiment classification
- **Named Entity Recognition**: Extract companies, products, people
- **Relationship Extraction**: Identify causal relationships

### 3. Technical Analysis
- RSI, MACD, Bollinger Bands
- Moving averages (50-day, 200-day)
- Volume profile analysis
- Support/resistance levels

### 4. Risk Management
- Position sizing algorithms
- Volatility-based risk adjustment
- Correlation analysis
- Drawdown protection

### 5. Explainability
- Clear reasoning for each recommendation
- Agent decision traces
- Confidence breakdown by signal
- Rejection explanations

---

## 🧪 Testing & Validation

### Backtesting

```bash
# Run historical backtest
python tests/backtest.py --start 2024-01-01 --end 2024-12-31 --symbols AAPL,GOOGL,MSFT
```

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Test specific agent
pytest tests/test_sentiment_agent.py
```

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: % of profitable trades
- **Max Drawdown**: Worst peak-to-trough decline
- **Agent Agreement**: % of aligned signals

---

## 🎯 Roadmap

### Phase 1: Foundation ✅
- [x] Project setup
- [ ] Sentiment agents (FinBERT)
- [ ] LangGraph workflow
- [ ] RAG knowledge base

### Phase 2: Intelligence 🔄
- [ ] Technical analysis agent
- [ ] Fundamental analysis agent
- [ ] Risk/critic agent
- [ ] Decision engine

### Phase 3: Execution 📋
- [ ] Alpaca paper trading integration
- [ ] Risk guardrails
- [ ] Order management
- [ ] Execution logging

### Phase 4: Interface 📋
- [ ] Streamlit dashboard
- [ ] FastAPI backend
- [ ] Agent observability
- [ ] Performance analytics

### Phase 5: Production 📋
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Monitoring & alerting
- [ ] Documentation

---

## 📚 Resources

### Financial AI Models
- [FinBERT (ProsusAI)](https://huggingface.co/ProsusAI/finbert)
- [FinBERT-Tone](https://huggingface.co/yiyanghkust/finbert-tone)

### Agent Frameworks
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Guide](https://docs.llamaindex.ai/)

### Market Data APIs
- [Alpaca API](https://alpaca.markets/docs/)
- [Alpha Vantage](https://www.alphavantage.co/)
- [yfinance](https://github.com/ranaroussi/yfinance)

---

## ⚠️ Disclaimers

1. **Not Financial Advice**: This system is for educational and research purposes only
2. **Paper Trading Default**: All trades execute in simulation mode unless explicitly changed
3. **Risk Warning**: Past performance does not guarantee future results
4. **Responsible AI**: Always review recommendations before executing real trades

---

## 🤝 Contributing

This is a capstone/portfolio project. Contributions, suggestions, and feedback are welcome!

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Nitin Digraje**
- GitHub: [@nitindme](https://github.com/nitindme)
- LinkedIn: [Your LinkedIn]
- Portfolio: [Your Website]

---

## 🏆 Interview Talking Points

**For Technical Interviews:**
1. **LangGraph Orchestration**: Multi-agent conditional routing with state management
2. **Domain-Specific AI**: Fine-tuned financial sentiment models (FinBERT)
3. **RAG Architecture**: LlamaIndex for grounded, cited responses from financial documents
4. **Risk Engineering**: Production-grade guardrails and human-in-the-loop controls
5. **System Design**: Scalable architecture with async FastAPI, vector databases, and broker APIs

**Key Differentiators:**
- ✅ Real broker API integration (Alpaca)
- ✅ Multi-agent validation (not just one LLM)
- ✅ Explainable AI (reasoning traces)
- ✅ Safety-first design (guardrails, simulation mode)
- ✅ Production-ready code structure

---

**Status**: 🚧 Phase 1 - Foundation Setup
**Next**: Implement sentiment agents with FinBERT
