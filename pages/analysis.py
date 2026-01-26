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
            sentiment = result.get('sentiment', {})
            
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
            st.info(reasoning)
        
        # TECHNICAL TAB
        with tab2:
            technical = result.get('technical', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Trend")
                trend = technical.get('trend', 'NEUTRAL')
                strength = technical.get('strength', 0)
                
                trend_emoji = {
                    "BULLISH": "🟢",
                    "BEARISH": "🔴",
                    "NEUTRAL": "🟡"
                }
                
                st.markdown(f"## {trend_emoji.get(trend, '⚪')} {trend}")
                st.progress(strength, text=f"Strength: {strength:.1%}")
            
            with col2:
                st.markdown("### RSI")
                rsi = technical.get('rsi', {}).get('value', 50)
                rsi_signal = technical.get('rsi', {}).get('signal', 'neutral')
                
                st.metric("RSI Value", f"{rsi:.1f}", rsi_signal.upper())
                
                if rsi > 70:
                    st.warning("⚠️ Overbought")
                elif rsi < 30:
                    st.success("✅ Oversold")
                else:
                    st.info("📊 Neutral")
            
            with col3:
                st.markdown("### MACD")
                macd = technical.get('macd', {})
                macd_signal = macd.get('signal', 'neutral')
                trend_macd = macd.get('trend', 'neutral')
                
                st.metric("MACD Signal", macd_signal.upper())
                st.markdown(f"**Trend**: {trend_macd}")
            
            st.markdown("---")
            
            # Technical indicators details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Moving Averages")
                ma = technical.get('moving_averages', {})
                
                sma_50 = ma.get('sma_50', 0)
                sma_200 = ma.get('sma_200', 0)
                signal_ma = ma.get('signal', 'neutral')
                
                st.markdown(f"**SMA 50**: ${sma_50:.2f}")
                st.markdown(f"**SMA 200**: ${sma_200:.2f}")
                st.markdown(f"**Signal**: {signal_ma.upper()}")
                
                if sma_50 > sma_200:
                    st.success("✅ Golden Cross (Bullish)")
                elif sma_50 < sma_200:
                    st.warning("⚠️ Death Cross (Bearish)")
            
            with col2:
                st.markdown("### Volume & Bollinger")
                
                volume = technical.get('volume', {})
                vol_trend = volume.get('trend', 'stable')
                st.markdown(f"**Volume Trend**: {vol_trend.upper()}")
                
                bb = technical.get('bollinger_bands', {})
                bb_signal = bb.get('signal', 'neutral')
                st.markdown(f"**Bollinger Signal**: {bb_signal.upper()}")
            
            st.markdown("---")
            
            reasoning_tech = technical.get('reasoning', 'No reasoning available')
            st.markdown("### 📝 Technical Analysis")
            st.info(reasoning_tech)
        
        # FUNDAMENTAL TAB
        with tab3:
            fundamental = result.get('fundamental', {})
            
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
            
            categories = [
                ('valuation', 'Valuation', col1),
                ('growth', 'Growth', col2),
                ('profitability', 'Profitability', col3),
                ('financial_health', 'Financial Health', col4)
            ]
            
            for key, label, col in categories:
                with col:
                    category_data = fundamental.get(key, {})
                    score = category_data.get('score', 0)
                    
                    st.markdown(f"**{label}**")
                    st.progress(score, text=f"{score:.1%}")
                    
                    # Show key metrics
                    metrics = category_data.get('metrics', {})
                    for metric_key, metric_value in list(metrics.items())[:2]:
                        if isinstance(metric_value, (int, float)):
                            st.markdown(f"*{metric_key}*: {metric_value:.2f}")
            
            st.markdown("---")
            
            reasoning_fund = fundamental.get('reasoning', 'No reasoning available')
            st.markdown("### 📝 Fundamental Analysis")
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
            
            # Guardrail checks
            st.markdown("### 🛡️ Guardrail Checks")
            
            guardrails = risk.get('guardrails_passed', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                volatility_check = guardrails.get('volatility_check', False)
                liquidity_check = guardrails.get('liquidity_check', False)
                
                st.markdown("**Volatility Check**")
                if volatility_check:
                    st.success("✅ PASS - Volatility within limits")
                else:
                    st.error("❌ FAIL - Volatility too high")
                
                st.markdown("**Liquidity Check**")
                if liquidity_check:
                    st.success("✅ PASS - Sufficient liquidity")
                else:
                    st.error("❌ FAIL - Low liquidity")
            
            with col2:
                confidence_check = guardrails.get('confidence_check', False)
                conflict_check = guardrails.get('conflict_check', False)
                
                st.markdown("**Confidence Check**")
                if confidence_check:
                    st.success("✅ PASS - High confidence")
                else:
                    st.error("❌ FAIL - Low confidence")
                
                st.markdown("**Conflict Check**")
                if conflict_check:
                    st.success("✅ PASS - Signals aligned")
                else:
                    st.error("❌ FAIL - Conflicting signals")
            
            st.markdown("---")
            
            # Risk factors
            st.markdown("### ⚠️ Risk Factors")
            
            risk_factors = risk.get('risk_factors', [])
            if risk_factors:
                for factor in risk_factors:
                    st.warning(f"- {factor}")
            else:
                st.success("✅ No major risk factors identified")
            
            st.markdown("---")
            
            reasoning_risk = risk.get('reasoning', 'No reasoning available')
            st.markdown("### 📝 Risk Analysis")
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
