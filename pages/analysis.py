"""
Stock Analysis Page
Real-time multi-agent analysis with interactive charts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from graph.workflow import TradingWorkflow
from data.market_data import MarketDataProvider

def render():
    st.markdown("## 📊 Stock Analysis")
    st.markdown("### Multi-Agent Intelligence Analysis")
    st.markdown("---")
    
    # Input section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA, NVDA)"
        ).upper()
    
    with col2:
        mode = st.selectbox("Mode", ["SIMULATION", "LIVE"])
    
    with col3:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Analysis execution
    if analyze_btn and symbol:
        with st.spinner(f"🤖 Running multi-agent analysis on {symbol}..."):
            try:
                # Initialize workflow
                workflow = TradingWorkflow(
                    min_confidence=0.75,
                    max_volatility=0.30
                )
                
                # Run analysis
                result = workflow.analyze_stock(symbol, mode=mode)
                
                # Store in session state
                st.session_state['analysis_result'] = result
                st.session_state['analysis_symbol'] = symbol
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return
    
    # Display results if available
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        symbol = st.session_state['analysis_symbol']
        
        # Summary header
        st.markdown(f"### 📈 Analysis Results: {symbol}")
        
        # Get recommendation details
        recommendation = result.get('recommendation', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        trade_approved = result.get('trade_approved', False)
        risk_level = result.get('risk', {}).get('risk_level', 'UNKNOWN')
        
        # Color coding
        if recommendation == "BUY" and trade_approved:
            rec_color = "success"
            emoji = "🟢"
        elif recommendation == "SELL":
            rec_color = "danger"
            emoji = "🔴"
        elif recommendation == "HOLD":
            rec_color = "warning"
            emoji = "🟡"
        else:
            rec_color = "secondary"
            emoji = "⚪"
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="{rec_color}-card">', unsafe_allow_html=True)
            st.metric(
                label="Recommendation",
                value=f"{emoji} {recommendation}",
                delta="Approved" if trade_approved else "Rejected"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence:.1%}",
                delta="High" if confidence >= 0.75 else "Low"
            )
        
        with col3:
            risk_colors = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "EXTREME": "🔴"
            }
            st.metric(
                label="Risk Level",
                value=f"{risk_colors.get(risk_level, '⚪')} {risk_level}"
            )
        
        with col4:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.metric(
                label="Analysis Time",
                value=timestamp
            )
        
        st.markdown("---")
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "😊 Sentiment",
            "📈 Technical",
            "💰 Fundamental",
            "🛡️ Risk",
            "📊 Charts"
        ])
        
        # SENTIMENT TAB
        with tab1:
            sentiment = result.get('sentiment') or {}
            
            if not sentiment:
                st.warning("⚠️ Sentiment analysis data not available")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Sentiment Score")
                    
                    score = sentiment.get('score', 0)
                    label = sentiment.get('label', 'neutral')
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Sentiment: {label.upper()}"},
                        delta={'reference': 0.5},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-1, -0.3], 'color': "lightcoral"},
                                {'range': [-0.3, 0.3], 'color': "lightyellow"},
                                {'range': [0.3, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Confidence & Sources")
                    
                    confidence_sent = sentiment.get('confidence', 0)
                    st.progress(confidence_sent, text=f"Confidence: {confidence_sent:.1%}")
                    
                    sources = sentiment.get('sources', [])
                    st.markdown("**Data Sources:**")
                    for source in sources[:5]:
                        st.markdown(f"- {source}")
                
                st.markdown("---")
                
                reasoning = sentiment.get('reasoning', 'No reasoning available')
                st.markdown("### 📝 Analysis Reasoning")
                
                # Display reasoning as bullet points if it's a list
                if isinstance(reasoning, list):
                    for point in reasoning:
                        st.markdown(f"- {point}")
                else:
                    st.info(reasoning)
        
        # TECHNICAL TAB
        with tab2:
            technical = result.get('technical') or {}
            
            if not technical:
                st.warning("⚠️ Technical analysis data not available")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Trend")
                    trend = technical.get('overall_trend', 'NEUTRAL')
                    strength = technical.get('trend_strength', 0)
                    
                    trend_emoji = {
                        "BULLISH": "🟢",
                        "BEARISH": "🔴",
                        "NEUTRAL": "🟡"
                    }
                    
                    st.markdown(f"## {trend_emoji.get(trend, '⚪')} {trend}")
                    st.progress(strength, text=f"Strength: {strength:.1%}")
                
                with col2:
                    st.markdown("### RSI")
                    rsi = technical.get('rsi', 50)
                    rsi_signal = technical.get('rsi_signal', 'neutral')
                    
                    st.metric("RSI Value", f"{rsi:.1f}", rsi_signal.upper())
                    
                    if rsi > 70:
                        st.warning("⚠️ Overbought")
                    elif rsi < 30:
                        st.success("✅ Oversold")
                    else:
                        st.info("📊 Neutral")
                
                with col3:
                    st.markdown("### MACD")
                    macd_value = technical.get('macd', 0)
                    macd_signal_value = technical.get('macd_signal', 0)
                    trend_macd = technical.get('macd_trend', 'neutral')
                    
                    st.metric("MACD", f"{macd_value:.2f}")
                    st.metric("Signal", f"{macd_signal_value:.2f}")
                    st.markdown(f"**Trend**: {trend_macd.upper()}")
                
                st.markdown("---")
                
                # Technical indicators details - Row 1
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Moving Averages")
                    
                    sma_50 = technical.get('sma_50', 0)
                    sma_200 = technical.get('sma_200', 0)
                    signal_ma = technical.get('ma_signal', 'neutral')
                    
                    st.markdown(f"**SMA 50**: ${sma_50:.2f}")
                    st.markdown(f"**SMA 200**: ${sma_200:.2f}")
                    st.markdown(f"**Signal**: {signal_ma.upper()}")
                    
                    if sma_50 > sma_200:
                        st.success("✅ Golden Cross (Bullish)")
                    elif sma_50 < sma_200:
                        st.warning("⚠️ Death Cross (Bearish)")
                
                with col2:
                    st.markdown("### Stochastic Oscillator")
                    stoch_k = technical.get('stoch_k', 50)
                    stoch_d = technical.get('stoch_d', 50)
                    stoch_signal = technical.get('stoch_signal', 'NEUTRAL')
                    
                    st.metric("%K", f"{stoch_k:.1f}")
                    st.metric("%D", f"{stoch_d:.1f}")
                    
                    if stoch_signal == "OVERBOUGHT":
                        st.warning("⚠️ Overbought")
                    elif stoch_signal == "OVERSOLD":
                        st.success("✅ Oversold - Potential Buy")
                    else:
                        st.info("📊 Neutral")
                
                with col3:
                    st.markdown("### ADX (Trend Strength)")
                    adx = technical.get('adx', 0)
                    adx_signal = technical.get('adx_signal', 'NO_TREND')
                    plus_di = technical.get('plus_di', 0)
                    minus_di = technical.get('minus_di', 0)
                    
                    st.metric("ADX", f"{adx:.1f}", adx_signal)
                    st.markdown(f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f}")
                    
                    if adx_signal == "STRONG_TREND":
                        st.success("✅ Strong Trend")
                    elif adx_signal == "WEAK_TREND":
                        st.info("📊 Weak Trend")
                    else:
                        st.warning("⚠️ No Clear Trend")
                
                st.markdown("---")
                
                # Technical indicators - Row 2
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Bollinger Bands")
                    bb_upper = technical.get('bb_upper', 0)
                    bb_middle = technical.get('bb_middle', 0)
                    bb_lower = technical.get('bb_lower', 0)
                    bb_signal = technical.get('bb_signal', 'NEUTRAL')
                    
                    st.markdown(f"**Upper**: ${bb_upper:.2f}")
                    st.markdown(f"**Middle**: ${bb_middle:.2f}")
                    st.markdown(f"**Lower**: ${bb_lower:.2f}")
                    st.markdown(f"**Signal**: {bb_signal}")
                
                with col2:
                    st.markdown("### Volatility (ATR)")
                    atr = technical.get('atr', 0)
                    atr_percent = technical.get('atr_percent', 0)
                    
                    st.metric("ATR", f"${atr:.2f}")
                    st.metric("ATR %", f"{atr_percent:.2f}%")
                    
                    if atr_percent > 3:
                        st.error("🔴 High Volatility")
                    elif atr_percent > 2:
                        st.warning("🟡 Moderate Volatility")
                    else:
                        st.success("🟢 Low Volatility")
                
                with col3:
                    st.markdown("### Support/Resistance")
                    support = technical.get('support_level', 0)
                    resistance = technical.get('resistance_level', 0)
                    price_to_support = technical.get('price_to_support', 0)
                    price_to_resistance = technical.get('price_to_resistance', 0)
                    
                    st.markdown(f"**Support**: ${support:.2f} ({price_to_support:+.1f}%)")
                    st.markdown(f"**Resistance**: ${resistance:.2f} ({price_to_resistance:+.1f}%)")
                
                st.markdown("---")
                
                # Volume Analysis
                st.markdown("### Volume Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    vol_trend = technical.get('volume_trend', 'NORMAL')
                    volume = technical.get('volume', 0)
                    volume_sma = technical.get('volume_sma_20', 0)
                    
                    st.metric("Current Volume", f"{volume:,.0f}")
                    if volume_sma > 0:
                        vol_ratio = (volume / volume_sma - 1) * 100
                        st.metric("Volume vs Avg", f"{vol_ratio:+.1f}%")
                
                with col2:
                    st.markdown(f"**Trend**: {vol_trend}")
                    if vol_trend == "INCREASING":
                        st.success("📈 High volume confirms trend")
                    elif vol_trend == "DECREASING":
                        st.warning("📉 Low volume - weak conviction")
                
                st.markdown("---")
                
                reasoning_tech = technical.get('reasoning', 'No reasoning available')
                st.markdown("### 📝 Technical Analysis")
                
                # Display reasoning as bullet points if it's a list
                if isinstance(reasoning_tech, list):
                    for point in reasoning_tech:
                        st.markdown(f"- {point}")
                else:
                    st.info(reasoning_tech)
        
        # FUNDAMENTAL TAB
        with tab3:
            fundamental = result.get('fundamental') or {}
            
            if not fundamental:
                st.warning("⚠️ Fundamental analysis data not available")
            else:
                overall_score = fundamental.get('overall_score', 0)
                overall_signal = fundamental.get('overall_signal', 'HOLD')
                
                # Overall score gauge
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=overall_score * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Fundamental Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightcoral"},
                                {'range': [40, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Investment Signal")
                    st.markdown(f"## {overall_signal}")
                    
                    signal_colors = {
                        "STRONG_BUY": "success",
                        "BUY": "info",
                        "HOLD": "warning",
                        "AVOID": "error"
                    }
                    
                    color = signal_colors.get(overall_signal, "secondary")
                    st.markdown(f"**Recommendation**: :{color}[{overall_signal}]")
                
                st.markdown("---")
                
                # Category scores
                st.markdown("### 📊 Category Breakdown")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Valuation
                with col1:
                    st.markdown("**Valuation**")
                    val_score = fundamental.get('valuation_score', 0)
                    val_signal = fundamental.get('valuation_signal', 'FAIR')
                    st.progress(val_score, text=f"{val_score:.1%}")
                    st.markdown(f"*Signal*: {val_signal}")
                    
                    pe = fundamental.get('pe_ratio')
                    if pe:
                        st.markdown(f"*P/E*: {pe:.2f}")
                    
                    # NEW: Additional valuation metrics
                    ev_ebitda = fundamental.get('ev_to_ebitda')
                    if ev_ebitda:
                        st.markdown(f"*EV/EBITDA*: {ev_ebitda:.2f}")
                    
                    ps = fundamental.get('price_to_sales')
                    if ps:
                        st.markdown(f"*P/S*: {ps:.2f}")
                
                # Growth
                with col2:
                    st.markdown("**Growth**")
                    growth_score = fundamental.get('growth_score', 0)
                    growth_signal = fundamental.get('growth_signal', 'STABLE')
                    st.progress(growth_score, text=f"{growth_score:.1%}")
                    st.markdown(f"*Signal*: {growth_signal}")
                    
                    rev_growth = fundamental.get('revenue_growth')
                    if rev_growth:
                        st.markdown(f"*Revenue*: {rev_growth:.2%}")
                    
                    earn_growth = fundamental.get('earnings_growth')
                    if earn_growth:
                        st.markdown(f"*Earnings*: {earn_growth:.2%}")
                
                # Profitability
                with col3:
                    st.markdown("**Profitability**")
                    prof_score = fundamental.get('profitability_score', 0)
                    prof_signal = fundamental.get('profitability_signal', 'MODERATE')
                    st.progress(prof_score, text=f"{prof_score:.1%}")
                    st.markdown(f"*Signal*: {prof_signal}")
                    
                    margin = fundamental.get('profit_margin')
                    if margin:
                        st.markdown(f"*Margin*: {margin:.2%}")
                    
                    roe = fundamental.get('roe')
                    if roe:
                        st.markdown(f"*ROE*: {roe:.2%}")
                    
                    roa = fundamental.get('roa')
                    if roa:
                        st.markdown(f"*ROA*: {roa:.2%}")
                
                # Financial Health
                with col4:
                    st.markdown("**Financial Health**")
                    health_score = fundamental.get('health_score', 0)
                    health_signal = fundamental.get('health_signal', 'STABLE')
                    st.progress(health_score, text=f"{health_score:.1%}")
                    st.markdown(f"*Signal*: {health_signal}")
                    
                    debt_ratio = fundamental.get('debt_to_equity')
                    if debt_ratio:
                        st.markdown(f"*D/E*: {debt_ratio:.2f}")
                    
                    current_ratio = fundamental.get('current_ratio')
                    if current_ratio:
                        st.markdown(f"*Current*: {current_ratio:.2f}")
                
                st.markdown("---")
                
                # NEW: Additional Fundamental Metrics
                st.markdown("### 💵 Shareholder Returns & Efficiency")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Dividends**")
                    div_yield = fundamental.get('dividend_yield')
                    payout = fundamental.get('payout_ratio')
                    
                    if div_yield:
                        st.metric("Dividend Yield", f"{div_yield:.2%}")
                    else:
                        st.markdown("*No dividend*")
                    
                    if payout:
                        st.markdown(f"*Payout Ratio*: {payout:.1%}")
                
                with col2:
                    st.markdown("**Cash Flow**")
                    fcf = fundamental.get('free_cash_flow')
                    if fcf:
                        if fcf >= 1_000_000_000:
                            st.metric("Free Cash Flow", f"${fcf/1e9:.1f}B")
                        elif fcf >= 1_000_000:
                            st.metric("Free Cash Flow", f"${fcf/1e6:.1f}M")
                        else:
                            st.metric("Free Cash Flow", f"${fcf:,.0f}")
                    else:
                        st.markdown("*FCF not available*")
                
                with col3:
                    st.markdown("**Efficiency**")
                    asset_turn = fundamental.get('asset_turnover')
                    gross_margin = fundamental.get('gross_margin')
                    
                    if asset_turn:
                        st.markdown(f"*Asset Turnover*: {asset_turn:.2f}x")
                    if gross_margin:
                        st.markdown(f"*Gross Margin*: {gross_margin:.1%}")
                
                st.markdown("---")
                
                reasoning_fund = fundamental.get('reasoning', 'No reasoning available')
                st.markdown("### 📝 Fundamental Analysis")
                
                # Display reasoning as bullet points if it's a list
                if isinstance(reasoning_fund, list):
                    for point in reasoning_fund:
                        st.markdown(f"- {point}")
                else:
                    st.info(reasoning_fund)
        
        # RISK TAB
        with tab4:
            risk = result.get('risk', {})
            
            # Risk level header
            risk_level = risk.get('risk_level', 'UNKNOWN')
            trade_approved = risk.get('trade_approved', False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Risk Assessment")
                
                risk_color_map = {
                    "LOW": "success",
                    "MEDIUM": "warning",
                    "HIGH": "error",
                    "EXTREME": "error"
                }
                
                color = risk_color_map.get(risk_level, "secondary")
                st.markdown(f"## :{color}[{risk_level}]")
                
                if trade_approved:
                    st.success("✅ TRADE APPROVED")
                else:
                    st.error("❌ TRADE REJECTED")
            
            with col2:
                st.markdown("### Overall Confidence")
                overall_conf = risk.get('overall_confidence', 0)
                
                st.progress(overall_conf, text=f"{overall_conf:.1%}")
                
                if overall_conf >= 0.75:
                    st.success("High Confidence")
                elif overall_conf >= 0.5:
                    st.warning("Medium Confidence")
                else:
                    st.error("Low Confidence")
            
            st.markdown("---")
            
            # NEW: Advanced Risk Metrics
            st.markdown("### 📊 Advanced Risk Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                beta = risk.get('beta')
                if beta is not None:
                    st.metric("Beta", f"{beta:.2f}")
                    if beta > 1.5:
                        st.caption("🔴 High market sensitivity")
                    elif beta < 0.5:
                        st.caption("🟢 Defensive stock")
                    else:
                        st.caption("🟡 Market-average")
                else:
                    st.metric("Beta", "N/A")
            
            with col2:
                sharpe = risk.get('sharpe_ratio')
                if sharpe is not None:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    if sharpe > 1:
                        st.caption("🟢 Good risk-adjusted return")
                    elif sharpe < 0:
                        st.caption("🔴 Negative returns")
                    else:
                        st.caption("🟡 Low risk-adjusted return")
                else:
                    st.metric("Sharpe Ratio", "N/A")
            
            with col3:
                max_dd = risk.get('max_drawdown')
                if max_dd is not None:
                    st.metric("Max Drawdown", f"{max_dd:.1%}")
                    if max_dd > 0.30:
                        st.caption("🔴 High drawdown risk")
                    elif max_dd > 0.20:
                        st.caption("🟡 Moderate drawdown")
                    else:
                        st.caption("🟢 Low drawdown")
                else:
                    st.metric("Max Drawdown", "N/A")
            
            with col4:
                var = risk.get('value_at_risk')
                if var is not None:
                    st.metric("Daily VaR (95%)", f"{var:.1%}")
                    st.caption("Max daily loss at 95% conf.")
                else:
                    st.metric("Daily VaR", "N/A")
            
            st.markdown("---")
            
            # NEW: Position Sizing & Trade Levels
            st.markdown("### � Position Sizing & Trade Levels")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                position_risk = risk.get('position_risk', 'MEDIUM')
                suggested_size = risk.get('suggested_position_size')
                
                st.markdown("**Position Risk**")
                risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                st.markdown(f"{risk_colors.get(position_risk, '⚪')} {position_risk}")
                
                if suggested_size:
                    st.metric("Suggested Size", f"{suggested_size:.1f}% of portfolio")
            
            with col2:
                stop_loss = risk.get('stop_loss_price')
                take_profit = risk.get('take_profit_price')
                
                if stop_loss:
                    st.metric("Stop Loss", f"${stop_loss:.2f}")
                if take_profit:
                    st.metric("Take Profit", f"${take_profit:.2f}")
            
            with col3:
                risk_reward = risk.get('risk_reward_ratio')
                sortino = risk.get('sortino_ratio')
                
                if risk_reward:
                    st.metric("Risk/Reward", f"{risk_reward:.1f}:1")
                    if risk_reward >= 2:
                        st.caption("🟢 Favorable R/R")
                    else:
                        st.caption("🟡 Marginal R/R")
                
                if sortino is not None:
                    st.metric("Sortino Ratio", f"{sortino:.2f}")
            
            st.markdown("---")
            
            # Guardrail checks
            st.markdown("### 🛡️ Guardrail Checks")
            
            col1, col2 = st.columns(2)
            
            with col1:
                volatility_check = risk.get('passes_volatility_check', False)
                liquidity_check = risk.get('passes_liquidity_check', False)
                
                st.markdown("**Volatility Check**")
                volatility_val = risk.get('volatility_value', 0)
                if volatility_check:
                    st.success(f"✅ PASS - {volatility_val:.1%} volatility")
                else:
                    st.error(f"❌ FAIL - {volatility_val:.1%} too high")
                
                st.markdown("**Liquidity Check**")
                volume = risk.get('volume', 0)
                if liquidity_check:
                    st.success(f"✅ PASS - {volume:,} volume")
                else:
                    st.error(f"❌ FAIL - {volume:,} insufficient")
            
            with col2:
                confidence_check = risk.get('passes_confidence_check', False)
                conflict_check = risk.get('passes_conflict_check', False)
                
                st.markdown("**Confidence Check**")
                if confidence_check:
                    st.success("✅ PASS - High confidence")
                else:
                    st.error("❌ FAIL - Low confidence")
                
                st.markdown("**Conflict Check**")
                if conflict_check:
                    st.success("✅ PASS - Signals aligned")
                else:
                    conflicts = risk.get('conflicting_signals', [])
                    st.error(f"❌ FAIL - {len(conflicts)} conflicts")
            
            st.markdown("---")
            
            # Risk factors
            st.markdown("### ⚠️ Risk Factors")
            
            risk_factors = risk.get('risk_factors', [])
            if risk_factors:
                for factor in risk_factors:
                    st.warning(f"⚠️ {factor}")
            else:
                st.success("✅ No major risk factors identified")
            
            st.markdown("---")
            
            reasoning_risk = risk.get('reasoning', 'No reasoning available')
            st.markdown("### 📝 Risk Analysis")
            
            # Display reasoning as bullet points if it's a list
            if isinstance(reasoning_risk, list):
                for point in reasoning_risk:
                    st.markdown(f"- {point}")
            else:
                st.info(reasoning_risk)
        
        # CHARTS TAB
        with tab5:
            st.markdown("### 📊 Price & Volume Charts")
            
            with st.spinner("Loading historical data..."):
                try:
                    # Fetch historical data
                    market_data = MarketDataProvider()
                    
                    # Get 6 months of data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=180)
                    
                    hist_data = market_data.data_source.history(
                        period="6mo",
                        interval="1d"
                    )
                    
                    if hist_data is not None and not hist_data.empty:
                        # Price chart with moving averages
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f'{symbol} Price & Moving Averages', 'Volume')
                        )
                        
                        # Candlestick
                        fig.add_trace(
                            go.Candlestick(
                                x=hist_data.index,
                                open=hist_data['Open'],
                                high=hist_data['High'],
                                low=hist_data['Low'],
                                close=hist_data['Close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )
                        
                        # Moving averages
                        if len(hist_data) >= 50:
                            ma_50 = hist_data['Close'].rolling(window=50).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=ma_50,
                                    name='SMA 50',
                                    line=dict(color='orange', width=2)
                                ),
                                row=1, col=1
                            )
                        
                        if len(hist_data) >= 200:
                            ma_200 = hist_data['Close'].rolling(window=200).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=ma_200,
                                    name='SMA 200',
                                    line=dict(color='blue', width=2)
                                ),
                                row=1, col=1
                            )
                        
                        # Volume
                        colors = ['red' if row['Open'] > row['Close'] else 'green' 
                                 for idx, row in hist_data.iterrows()]
                        
                        fig.add_trace(
                            go.Bar(
                                x=hist_data.index,
                                y=hist_data['Volume'],
                                name='Volume',
                                marker_color=colors
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            height=600,
                            xaxis_rangeslider_visible=False,
                            showlegend=True
                        )
                        
                        fig.update_xaxes(title_text="Date", row=2, col=1)
                        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig.update_yaxes(title_text="Volume", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # RSI chart
                        st.markdown("### 📈 RSI Indicator")
                        
                        # Calculate RSI
                        delta = hist_data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        fig_rsi = go.Figure()
                        
                        fig_rsi.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=rsi,
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ))
                        
                        # Add overbought/oversold lines
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                         annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                         annotation_text="Oversold")
                        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
                        
                        fig_rsi.update_layout(
                            height=300,
                            yaxis=dict(range=[0, 100]),
                            xaxis_title="Date",
                            yaxis_title="RSI"
                        )
                        
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    else:
                        st.warning("Could not load historical chart data")
                
                except Exception as e:
                    st.error(f"Error loading charts: {str(e)}")
    
    else:
        # Placeholder when no analysis
        st.info("👆 Enter a stock symbol and click **Analyze** to get started")
        
        st.markdown("### 💡 Example Stocks to Try:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Tech Giants**")
            st.markdown("- AAPL (Apple)")
            st.markdown("- MSFT (Microsoft)")
            st.markdown("- GOOGL (Google)")
        
        with col2:
            st.markdown("**EV & AI**")
            st.markdown("- TSLA (Tesla)")
            st.markdown("- NVDA (NVIDIA)")
            st.markdown("- AMD (AMD)")
        
        with col3:
            st.markdown("**Finance**")
            st.markdown("- JPM (JP Morgan)")
            st.markdown("- GS (Goldman Sachs)")
            st.markdown("- V (Visa)")
        
        with col4:
            st.markdown("**Retail**")
            st.markdown("- AMZN (Amazon)")
            st.markdown("- WMT (Walmart)")
            st.markdown("- TGT (Target)")
