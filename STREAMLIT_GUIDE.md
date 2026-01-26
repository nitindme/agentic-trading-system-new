# 🎨 Streamlit UI Guide - Agentic Trading System

## 📋 Overview

This guide will help you launch and use the **interactive Streamlit dashboard** for the Agentic Trading System. The UI provides real-time stock analysis, portfolio tracking, risk assessment, and visual charts.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/nitindigraje/Documents/agentic-trading-system

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install all dependencies (including Streamlit & Plotly)
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

### 3. Open in Browser

The dashboard will automatically open at **http://localhost:8501**

---

## 🎯 Dashboard Features

### 🏠 Home Page

**What it shows:**
- System overview with 5 AI agents
- Feature highlights (Sentiment, Technical, Fundamental, Risk)
- Quick stats (agents, indicators, confidence thresholds)
- How the multi-agent workflow operates
- Getting started guide

**Use case:**
- First-time users to understand the system
- Quick reference for system capabilities

---

### 📊 Stock Analysis Page

**Features:**
1. **Real-time Analysis**
   - Enter any stock symbol (AAPL, TSLA, NVDA, etc.)
   - Run complete multi-agent analysis
   - Get BUY/HOLD/SELL recommendation with confidence

2. **Interactive Tabs**
   - 😊 **Sentiment**: FinBERT score, confidence gauge, reasoning
   - 📈 **Technical**: RSI, MACD, Moving Averages, Bollinger Bands
   - 💰 **Fundamental**: Valuation, growth, profitability, health scores
   - 🛡️ **Risk**: Guardrail checks, trade approval decision
   - 📊 **Charts**: Candlestick price chart, volume, RSI indicator

3. **Visual Elements**
   - Gauge charts for sentiment scores
   - Progress bars for confidence levels
   - Candlestick charts with moving averages
   - RSI indicator with overbought/oversold zones
   - Color-coded recommendations (🟢 BUY, 🟡 HOLD, 🔴 SELL)

**How to use:**
```
1. Navigate to "📊 Stock Analysis" in sidebar
2. Enter stock symbol (e.g., "NVDA")
3. Select mode (SIMULATION or LIVE)
4. Click "🔍 Analyze"
5. Explore results in tabs
6. View price charts and technical indicators
```

**Example workflow:**
- Analyzing AAPL → Gets 82% confidence BUY signal
- Technical shows BULLISH trend with RSI at 58
- Fundamental shows strong profitability (ROE > 15%)
- Risk agent approves trade (all guardrails pass)

---

### 💼 Portfolio Page

**Features:**
1. **Watchlist Management**
   - Add stocks to your watchlist
   - Remove stocks easily
   - Default watchlist: AAPL, MSFT, GOOGL, TSLA

2. **Batch Analysis**
   - Click "🔄 Analyze All" to run multi-agent analysis on entire watchlist
   - View results in real-time table
   - Compare recommendations side-by-side

3. **Portfolio Analytics**
   - Total stocks analyzed
   - Buy signals count and percentage
   - Approved trades count
   - Average confidence across portfolio

4. **Visualizations**
   - 📈 Recommendation distribution (pie chart)
   - 🛡️ Risk distribution (bar chart)
   - 📊 Confidence comparison (grouped bar chart)
   - 📋 Detailed comparison table

5. **Export Functionality**
   - Export portfolio analysis to CSV
   - Includes timestamp, recommendations, confidence, risk levels
   - Download button generates CSV file

**How to use:**
```
1. Navigate to "💼 Portfolio" in sidebar
2. Add stocks to watchlist using text input + "➕ Add"
3. Click "🔄 Analyze All" to run analysis
4. View analytics and charts
5. Export to CSV for external analysis
6. Remove stocks with "🗑️ Remove" button
```

**Use cases:**
- Track multiple stocks simultaneously
- Compare risk levels across portfolio
- Identify best opportunities (highest confidence + approved)
- Export data for backtesting or reporting

---

### 🛡️ Risk Dashboard

**Features:**
1. **Risk Parameter Configuration**
   - Minimum confidence threshold (default: 75%)
   - Maximum volatility threshold (default: 30%)
   - Minimum daily volume (default: 1M)
   - Maximum signal conflicts (default: 30%)

2. **Individual Stock Risk Analysis**
   - Enter symbol for detailed risk assessment
   - Real-time guardrail checks
   - Risk level determination (LOW/MEDIUM/HIGH/EXTREME)
   - Trade approval/rejection decision

3. **Guardrail Monitoring**
   - ✅ Volatility Check: 30-day volatility ≤ threshold
   - ✅ Liquidity Check: Daily volume ≥ minimum
   - ✅ Confidence Check: Overall confidence ≥ minimum
   - ✅ Conflict Check: Agent disagreements ≤ maximum

4. **Visual Risk Assessment**
   - Progress bars for each guardrail
   - Color-coded pass/fail indicators
   - Guardrail summary chart
   - Confidence breakdown (Sentiment, Technical, Fundamental)
   - Portfolio risk heatmap

5. **Risk Factors**
   - Detailed list of identified risks
   - Reasoning behind risk assessment
   - Recommendations for risk mitigation

**How to use:**
```
1. Navigate to "🛡️ Risk Dashboard" in sidebar
2. Adjust risk parameters using sliders
3. Enter stock symbol
4. Click "🔍 Analyze Risk"
5. Review guardrail checks
6. Read risk factors and reasoning
7. View portfolio risk heatmap (if portfolio analyzed)
```

**Use cases:**
- Fine-tune risk tolerance for your strategy
- Understand why trades are rejected
- Monitor guardrail effectiveness
- Compare risk across multiple stocks

---

### ⚙️ Settings Page

**Features:**
1. **Agent Configuration**
   - Adjust confidence thresholds
   - Set volatility limits
   - Configure volume minimums
   - Set conflict tolerance

2. **Model Settings**
   - Select sentiment model (FinBERT, Twitter-RoBERTa)
   - Enable/disable ensemble sentiment
   - Choose primary data source

3. **Data Sources**
   - Select market data provider (yfinance, Alpha Vantage)
   - Enable/disable news sentiment
   - Enable/disable social sentiment

4. **System Information**
   - Version info
   - Python version
   - Active agents count
   - Model status

**How to use:**
```
1. Navigate to "⚙️ Settings" in sidebar
2. Adjust parameters as needed
3. Click "💾 Save Configuration"
4. Restart app if required
```

---

## 🎨 UI Components Reference

### Color Coding

**Recommendations:**
- 🟢 **Green (Success)**: BUY signals, approved trades
- 🟡 **Yellow (Warning)**: HOLD signals, medium risk
- 🔴 **Red (Danger)**: SELL signals, rejected trades
- ⚪ **Gray (Neutral)**: Unknown or neutral states

**Risk Levels:**
- 🟢 **LOW**: All guardrails pass, proceed with confidence
- 🟡 **MEDIUM**: Minor concerns, acceptable risk
- 🟠 **HIGH**: Multiple warnings, proceed with caution
- 🔴 **EXTREME**: Unacceptable risk, trade rejected

### Charts & Visualizations

**Gauge Charts:**
- Sentiment score visualization (-1 to +1)
- Fundamental score (0 to 100)
- Visual threshold markers

**Candlestick Charts:**
- OHLC (Open, High, Low, Close) price data
- 50-day and 200-day moving averages
- Volume bars (green = up day, red = down day)

**RSI Indicator:**
- 14-period RSI line
- Overbought zone (>70) in red
- Oversold zone (<30) in green
- Neutral line at 50

**Bar Charts:**
- Confidence comparison across stocks
- Risk distribution
- Guardrail status
- Weighted confidence breakdown

**Pie Charts:**
- Recommendation distribution
- Confidence component breakdown

**Heatmaps:**
- Portfolio risk matrix
- Multiple dimensions: volatility, liquidity, confidence, risk score

---

## 💡 Pro Tips

### Performance Optimization

1. **Use Simulation Mode**
   - Faster for testing
   - Doesn't hit live APIs
   - Good for UI exploration

2. **Batch Analysis**
   - Analyze multiple stocks at once in Portfolio page
   - More efficient than individual analyses

3. **Cache Results**
   - Analysis results stored in session state
   - No need to re-analyze unless data changes

### Best Practices

1. **Start with Home Page**
   - Understand system capabilities first
   - Review how agents work together

2. **Test Individual Stocks**
   - Use Stock Analysis page first
   - Understand what each agent contributes
   - Review all tabs for complete picture

3. **Build Watchlist Gradually**
   - Add 3-5 stocks initially
   - Analyze and review results
   - Expand watchlist based on insights

4. **Monitor Risk Dashboard**
   - Check guardrail settings match your risk tolerance
   - Review rejected trades to understand why
   - Use risk heatmap to identify problematic stocks

5. **Export Data Regularly**
   - Export portfolio analyses to CSV
   - Track recommendations over time
   - Build historical performance data

### Troubleshooting

**Dashboard won't start:**
```bash
# Check if port 8501 is in use
lsof -i :8501

# Kill existing process if needed
kill -9 <PID>

# Try different port
streamlit run streamlit_app.py --server.port 8502
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check virtual environment
which python
```

**Charts not displaying:**
- Ensure plotly installed: `pip install plotly==5.18.0`
- Clear browser cache
- Try different browser (Chrome recommended)

**Analysis fails:**
- Check internet connection (for market data APIs)
- Verify stock symbol is valid
- Review error message in red box

**Slow performance:**
- Close unused tabs
- Clear session state (refresh page)
- Reduce watchlist size
- Use simulation mode

---

## 🔧 Customization

### Modify Styling

Edit custom CSS in `streamlit_app.py`:

```python
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;  # Change header size
        color: #1f77b4;   # Change header color
    }
    # Add your custom styles
</style>
""", unsafe_allow_html=True)
```

### Add New Pages

1. Create new file in `pages/` directory:
```python
# pages/my_custom_page.py

import streamlit as st

def render():
    st.markdown("## My Custom Page")
    # Your content here
```

2. Import in `streamlit_app.py`:
```python
from pages import analysis, portfolio, risk, my_custom_page

# Add to navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Stock Analysis", "💼 Portfolio", 
     "🛡️ Risk Dashboard", "🆕 My Custom Page", "⚙️ Settings"]
)

# Add page rendering
elif page == "🆕 My Custom Page":
    my_custom_page.render()
```

### Modify Agent Parameters

Edit default values in pages:

```python
# pages/analysis.py

workflow = TradingWorkflow(
    min_confidence=0.80,      # Raise threshold
    max_volatility=0.25,      # Lower volatility limit
    min_volume=2_000_000,     # Require more liquidity
    max_conflict_threshold=0.20  # Less tolerance for conflicts
)
```

---

## 📊 Sample Use Cases

### Use Case 1: Daily Market Scan

**Goal:** Identify top trading opportunities each morning

```
1. Open dashboard → Portfolio page
2. Add 10-15 stocks to watchlist (mix of sectors)
3. Click "🔄 Analyze All"
4. Sort by confidence (descending)
5. Focus on BUY signals with ✅ approved status
6. Review top 3 candidates in Stock Analysis page
7. Export results to CSV for record-keeping
```

### Use Case 2: Risk Assessment Before Trade

**Goal:** Validate trade decision with risk analysis

```
1. Navigate to Stock Analysis page
2. Enter stock symbol (e.g., TSLA)
3. Click "🔍 Analyze"
4. Review recommendation and confidence
5. Switch to Risk Dashboard
6. Enter same symbol
7. Review all 4 guardrail checks
8. Read risk factors
9. Make informed decision based on risk level
```

### Use Case 3: Portfolio Risk Monitoring

**Goal:** Monitor risk exposure across holdings

```
1. Add all portfolio holdings to watchlist
2. Run batch analysis
3. Navigate to Risk Dashboard
4. View portfolio risk heatmap
5. Identify high-risk positions (red zones)
6. Review individual risk assessments
7. Consider rebalancing based on risk distribution
```

### Use Case 4: Backtesting Strategy

**Goal:** Test strategy on historical stocks

```
1. Build watchlist of stocks from specific period
2. Analyze all stocks
3. Export results to CSV
4. Compare recommendations with actual outcomes
5. Calculate win rate and average returns
6. Adjust agent parameters in Settings
7. Re-analyze to see impact on recommendations
```

---

## 🎓 Learning the System

### Beginner Path (30 minutes)

1. **Home Page** (5 min)
   - Read system overview
   - Understand how 5 agents work together

2. **Stock Analysis** (15 min)
   - Analyze AAPL
   - Go through all 5 tabs
   - Understand what each agent contributes
   - View price charts

3. **Risk Dashboard** (10 min)
   - Analyze same stock (AAPL)
   - See how guardrails work
   - Understand approval/rejection logic

### Intermediate Path (1 hour)

1. **Portfolio Setup** (20 min)
   - Build watchlist of 5-7 stocks
   - Run batch analysis
   - Compare recommendations

2. **Deep Dive Analysis** (20 min)
   - Pick most interesting stock from portfolio
   - Review all tabs in Stock Analysis
   - Cross-reference with Risk Dashboard
   - Understand reasoning for recommendation

3. **Risk Tuning** (20 min)
   - Adjust risk parameters in Settings
   - Re-analyze same stocks
   - See how parameters affect approvals
   - Find your risk tolerance

### Advanced Path (2+ hours)

1. **Strategy Development**
   - Define trading strategy (e.g., "only BUY signals with 80%+ confidence and LOW risk")
   - Build watchlist of 20+ stocks
   - Batch analyze
   - Filter based on strategy rules

2. **Comparative Analysis**
   - Analyze competing stocks (AAPL vs MSFT)
   - Compare all metrics side-by-side
   - Use portfolio analytics

3. **Parameter Optimization**
   - Test different confidence thresholds
   - Compare approval rates
   - Find optimal risk settings
   - Document results

---

## 🚀 Next Steps

After mastering the UI:

1. **Phase 3 Integration**
   - Connect to Alpaca paper trading
   - Execute approved trades automatically
   - Track actual performance

2. **Custom Indicators**
   - Add more technical indicators
   - Implement custom scoring algorithms
   - Extend fundamental analysis

3. **Backtesting Module**
   - Build historical analysis capability
   - Calculate strategy performance metrics
   - Optimize agent parameters

4. **Production Deployment**
   - Deploy on Streamlit Cloud
   - Share with others
   - Showcase in portfolio

---

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Charts**: https://plotly.com/python/
- **Project README**: [README.md](./README.md)
- **Architecture**: [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Phase 2 Complete**: [PHASE2_COMPLETE.md](./PHASE2_COMPLETE.md)

---

## ❓ FAQ

**Q: Can I analyze any stock?**
A: Yes! Enter any valid ticker symbol (US markets). International stocks may have limited data.

**Q: How long does analysis take?**
A: 5-10 seconds per stock in SIMULATION mode. LIVE mode may take longer due to API calls.

**Q: Can I analyze crypto?**
A: Currently optimized for stocks. Crypto support planned for future releases.

**Q: Is data real-time?**
A: Uses yfinance which provides 15-20 minute delayed data. For real-time, upgrade to Alpha Vantage Premium.

**Q: Can I save my watchlist?**
A: Currently session-based. Watchlist clears on refresh. Persistence planned for Phase 3.

**Q: How accurate are recommendations?**
A: System provides analysis, not financial advice. Accuracy depends on market conditions and data quality. Always do your own research.

---

**Built with**: Streamlit 1.29 • Plotly 5.18 • LangGraph • FinBERT  
**Version**: 2.0.0 (Phase 2 Complete)  
**Author**: Nitin Digraje  
**License**: MIT
