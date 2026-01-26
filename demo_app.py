"""
Streamlit Dashboard for Agentic Trading System - Demo Mode
Simplified version for quick UI testing without heavy ML dependencies
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Agentic Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("# 🤖 Agentic Trading")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Demo Mode")
st.sidebar.info("Full analysis requires ML dependencies")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Available:")
st.sidebar.markdown("- ✅ Stock quotes (yfinance)")
st.sidebar.markdown("- ✅ Basic charts")
st.sidebar.markdown("- ✅ Price data")

# Main content
st.markdown('<p class="main-header">📊 Stock Quote Viewer</p>', unsafe_allow_html=True)
st.markdown("### Get Real-Time Stock Data")
st.markdown("---")

# Stock input
col1, col2 = st.columns([3, 1])

with col1:
    symbol = st.text_input(
        "Enter Stock Symbol",
        value="AAPL",
        help="Enter a valid stock ticker (e.g., AAPL, TSLA, MSFT, GOOGL)"
    ).upper()

with col2:
    get_quote = st.button("📊 Get Quote", type="primary", use_container_width=True)

st.markdown("---")

# Get stock data
if get_quote and symbol:
    try:
        import yfinance as yf
        
        with st.spinner(f"Fetching data for {symbol}..."):
            # Get stock data
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")
            
            if not hist.empty:
                # Display current price
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
                
                st.success(f"✅ Successfully retrieved data for {symbol}")
                st.markdown("---")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:.2f}",
                        delta=f"{price_change_pct:+.2f}%"
                    )
                
                with col2:
                    volume = hist['Volume'].iloc[-1]
                    st.metric(
                        label="Volume",
                        value=f"{volume:,.0f}"
                    )
                
                with col3:
                    high_price = hist['High'].iloc[-1]
                    st.metric(
                        label="Day High",
                        value=f"${high_price:.2f}"
                    )
                
                with col4:
                    low_price = hist['Low'].iloc[-1]
                    st.metric(
                        label="Day Low",
                        value=f"${low_price:.2f}"
                    )
                
                st.markdown("---")
                
                # Company info
                st.markdown("### 📋 Company Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'longName' in info:
                        st.markdown(f"**Company Name:** {info['longName']}")
                    if 'sector' in info:
                        st.markdown(f"**Sector:** {info['sector']}")
                    if 'industry' in info:
                        st.markdown(f"**Industry:** {info['industry']}")
                    if 'marketCap' in info:
                        market_cap = info['marketCap'] / 1e9
                        st.markdown(f"**Market Cap:** ${market_cap:.2f}B")
                
                with col2:
                    if 'trailingPE' in info and info['trailingPE']:
                        st.markdown(f"**P/E Ratio:** {info['trailingPE']:.2f}")
                    if 'dividendYield' in info and info['dividendYield']:
                        div_yield = info['dividendYield'] * 100
                        st.markdown(f"**Dividend Yield:** {div_yield:.2f}%")
                    if 'fiftyTwoWeekHigh' in info:
                        st.markdown(f"**52-Week High:** ${info['fiftyTwoWeekHigh']:.2f}")
                    if 'fiftyTwoWeekLow' in info:
                        st.markdown(f"**52-Week Low:** ${info['fiftyTwoWeekLow']:.2f}")
                
                st.markdown("---")
                
                # Price history
                st.markdown("### 📈 Recent Price History (5 Days)")
                
                # Display as table
                hist_display = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                hist_display['Date'] = hist_display.index.strftime('%Y-%m-%d')
                hist_display = hist_display[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                hist_display = hist_display.round(2)
                
                st.dataframe(
                    hist_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Try to create chart if plotly is available
                try:
                    import plotly.graph_objects as go
                    
                    st.markdown("---")
                    st.markdown("### 📊 Price Chart")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines+markers',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} - 5 Day Price Trend",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except ImportError:
                    st.info("💡 Install plotly for interactive charts: `pip install plotly`")
                
                # Quick analysis
                st.markdown("---")
                st.markdown("### 🤖 Quick Analysis")
                
                # Simple trend analysis
                if price_change_pct > 2:
                    st.success(f"📈 **Strong Upward Trend**: {symbol} is up {price_change_pct:.2f}% from previous close")
                elif price_change_pct > 0:
                    st.info(f"📊 **Slight Gain**: {symbol} is up {price_change_pct:.2f}% from previous close")
                elif price_change_pct < -2:
                    st.error(f"📉 **Strong Downward Trend**: {symbol} is down {price_change_pct:.2f}% from previous close")
                elif price_change_pct < 0:
                    st.warning(f"📊 **Slight Decline**: {symbol} is down {price_change_pct:.2f}% from previous close")
                else:
                    st.info(f"📊 **Stable**: {symbol} shows minimal price movement")
                
            else:
                st.error(f"❌ No data available for {symbol}")
    
    except Exception as e:
        st.error(f"❌ Error fetching data for {symbol}")
        st.error(f"Details: {str(e)}")
        st.info("💡 Make sure yfinance is installed: `pip install yfinance`")

else:
    # Instructions
    st.info("👆 Enter a stock symbol above and click **Get Quote** to view live data")
    
    st.markdown("### 💡 Popular Stocks to Try:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Tech Giants**")
        st.markdown("- AAPL (Apple)")
        st.markdown("- MSFT (Microsoft)")
        st.markdown("- GOOGL (Google)")
        st.markdown("- AMZN (Amazon)")
    
    with col2:
        st.markdown("**AI & Chips**")
        st.markdown("- NVDA (NVIDIA)")
        st.markdown("- AMD (AMD)")
        st.markdown("- INTC (Intel)")
        st.markdown("- TSM (TSMC)")
    
    with col3:
        st.markdown("**EV & Auto**")
        st.markdown("- TSLA (Tesla)")
        st.markdown("- F (Ford)")
        st.markdown("- GM (General Motors)")
        st.markdown("- RIVN (Rivian)")
    
    with col4:
        st.markdown("**Finance**")
        st.markdown("- JPM (JP Morgan)")
        st.markdown("- BAC (Bank of America)")
        st.markdown("- GS (Goldman Sachs)")
        st.markdown("- V (Visa)")
    
    st.markdown("---")
    st.markdown("### ℹ️ About This Demo")
    st.markdown("""
    This is a **simplified demo version** that shows:
    - ✅ Real-time stock quotes from Yahoo Finance
    - ✅ Company information and metrics
    - ✅ 5-day price history
    - ✅ Basic price charts (if plotly installed)
    - ✅ Simple trend analysis
    
    **For full multi-agent analysis** with sentiment, technical indicators, and fundamental scoring:
    - Install all dependencies: `pip install -r requirements.txt`
    - Use the full dashboard: `streamlit run streamlit_app.py`
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Agentic Trading System - Demo Mode | © 2026 Nitin Digraje</p>",
    unsafe_allow_html=True
)
