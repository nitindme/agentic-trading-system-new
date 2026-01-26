"""
Risk Dashboard Page
Comprehensive risk analysis and guardrail monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.risk_agent import RiskAgent
from data.market_data import MarketDataProvider

def render():
    st.markdown("## 🛡️ Risk Dashboard")
    st.markdown("### Comprehensive Risk Analysis & Guardrail Monitoring")
    st.markdown("---")
    
    # Risk configuration
    st.markdown("### ⚙️ Risk Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_confidence = st.slider(
            "Min Confidence",
            0.5, 1.0, 0.75, 0.05,
            help="Minimum confidence threshold for trade approval"
        )
    
    with col2:
        max_volatility = st.slider(
            "Max Volatility",
            0.1, 0.5, 0.30, 0.05,
            help="Maximum allowed 30-day volatility"
        )
    
    with col3:
        min_volume = st.number_input(
            "Min Volume",
            100000, 10000000, 1000000, 100000,
            help="Minimum daily trading volume"
        )
    
    with col4:
        max_conflicts = st.slider(
            "Max Conflicts",
            0.1, 0.5, 0.30, 0.05,
            help="Maximum allowed signal conflicts"
        )
    
    st.markdown("---")
    
    # Stock input for risk analysis
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "Analyze Risk for Stock",
            value="AAPL",
            help="Enter stock symbol for detailed risk analysis"
        ).upper()
    
    with col2:
        analyze_btn = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    if analyze_btn and symbol:
        with st.spinner(f"Analyzing risk for {symbol}..."):
            try:
                # Get market data
                market_data = MarketDataProvider()
                current_data = market_data.get_current_data(symbol)
                volatility = market_data.get_volatility(symbol)
                
                # Mock sentiment, technical, fundamental for risk assessment
                # In real scenario, these would come from the workflow
                mock_sentiment = {
                    'score': 0.65,
                    'confidence': 0.80,
                    'label': 'positive'
                }
                
                mock_technical = {
                    'trend': 'BULLISH',
                    'strength': 0.72,
                    'confidence': 0.75
                }
                
                mock_fundamental = {
                    'overall_score': 0.68,
                    'overall_signal': 'BUY'
                }
                
                # Initialize risk agent
                risk_agent = RiskAgent(
                    min_confidence=min_confidence,
                    max_volatility=max_volatility,
                    min_volume=min_volume,
                    max_conflict_threshold=max_conflicts
                )
                
                # Assess risk
                risk_assessment = risk_agent.assess_risk(
                    symbol=symbol,
                    sentiment=mock_sentiment,
                    technical=mock_technical,
                    fundamental=mock_fundamental,
                    market_data=current_data,
                    volatility=volatility
                )
                
                # Store in session state
                st.session_state['risk_analysis'] = risk_assessment
                st.session_state['risk_symbol'] = symbol
                
            except Exception as e:
                st.error(f"❌ Risk analysis failed: {str(e)}")
                return
    
    # Display risk analysis if available
    if 'risk_analysis' in st.session_state:
        risk = st.session_state['risk_analysis']
        symbol = st.session_state['risk_symbol']
        
        st.markdown(f"### 📊 Risk Analysis: {symbol}")
        
        # Risk level header
        risk_level = risk.risk_level
        trade_approved = risk.trade_approved
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_color_map = {
                "LOW": "success",
                "MEDIUM": "warning",
                "HIGH": "error",
                "EXTREME": "error"
            }
            
            risk_emoji = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "EXTREME": "🔴"
            }
            
            color = risk_color_map.get(risk_level, "secondary")
            emoji = risk_emoji.get(risk_level, "⚪")
            
            st.markdown(f"#### Risk Level")
            st.markdown(f"## {emoji} {risk_level}")
        
        with col2:
            st.markdown("#### Overall Confidence")
            st.metric(
                label="",
                value=f"{risk.overall_confidence:.1%}",
                delta="High" if risk.overall_confidence >= 0.75 else "Low"
            )
        
        with col3:
            st.markdown("#### Trade Decision")
            if trade_approved:
                st.success("✅ APPROVED")
            else:
                st.error("❌ REJECTED")
        
        with col4:
            st.markdown("#### Risk Factors")
            st.metric(
                label="",
                value=len(risk.risk_factors),
                delta="Issues Found" if len(risk.risk_factors) > 0 else "Clear"
            )
        
        st.markdown("---")
        
        # Guardrail checks visualization
        st.markdown("### 🛡️ Guardrail Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volatility check
            volatility_pass = risk.guardrails_passed.get('volatility_check', False)
            
            st.markdown("#### Volatility Check")
            if volatility_pass:
                st.success("✅ PASS - Volatility within acceptable limits")
            else:
                st.error("❌ FAIL - Volatility exceeds threshold")
            
            # Progress bar for volatility
            try:
                current_vol = risk.risk_factors[0].split(':')[1].strip() if 'volatility' in str(risk.risk_factors).lower() else "0%"
                vol_value = float(current_vol.strip('%')) / 100 if '%' in current_vol else 0
            except:
                vol_value = max_volatility * 0.8  # Default
            
            st.progress(min(vol_value / max_volatility, 1.0), 
                       text=f"Current: {vol_value:.1%} / Max: {max_volatility:.1%}")
            
            # Confidence check
            confidence_pass = risk.guardrails_passed.get('confidence_check', False)
            
            st.markdown("#### Confidence Check")
            if confidence_pass:
                st.success("✅ PASS - Confidence above minimum threshold")
            else:
                st.error("❌ FAIL - Confidence too low")
            
            st.progress(risk.overall_confidence, 
                       text=f"Current: {risk.overall_confidence:.1%} / Min: {min_confidence:.1%}")
        
        with col2:
            # Liquidity check
            liquidity_pass = risk.guardrails_passed.get('liquidity_check', False)
            
            st.markdown("#### Liquidity Check")
            if liquidity_pass:
                st.success("✅ PASS - Sufficient trading volume")
            else:
                st.error("❌ FAIL - Insufficient liquidity")
            
            # Mock current volume for visualization
            try:
                market_data = MarketDataProvider()
                current_data = market_data.get_current_data(symbol)
                current_volume = current_data.get('volume', min_volume)
            except:
                current_volume = min_volume * 1.5
            
            vol_ratio = min(current_volume / min_volume, 2.0)
            st.progress(vol_ratio / 2.0, 
                       text=f"Current: {current_volume:,.0f} / Min: {min_volume:,.0f}")
            
            # Conflict check
            conflict_pass = risk.guardrails_passed.get('conflict_check', False)
            
            st.markdown("#### Signal Conflict Check")
            if conflict_pass:
                st.success("✅ PASS - Agents aligned")
            else:
                st.error("❌ FAIL - Conflicting signals detected")
            
            # Mock conflict level
            conflict_level = 0.15 if conflict_pass else 0.35
            st.progress(conflict_level / max_conflicts, 
                       text=f"Conflict: {conflict_level:.1%} / Max: {max_conflicts:.1%}")
        
        st.markdown("---")
        
        # Guardrail summary chart
        st.markdown("### 📊 Guardrail Summary")
        
        guardrail_data = {
            'Check': ['Volatility', 'Liquidity', 'Confidence', 'Conflicts'],
            'Status': [
                risk.guardrails_passed.get('volatility_check', False),
                risk.guardrails_passed.get('liquidity_check', False),
                risk.guardrails_passed.get('confidence_check', False),
                risk.guardrails_passed.get('conflict_check', False)
            ]
        }
        
        df_guardrails = pd.DataFrame(guardrail_data)
        df_guardrails['Value'] = df_guardrails['Status'].apply(lambda x: 1 if x else 0)
        
        fig = go.Figure()
        
        colors = ['#28a745' if status else '#dc3545' for status in df_guardrails['Status']]
        
        fig.add_trace(go.Bar(
            x=df_guardrails['Check'],
            y=df_guardrails['Value'],
            marker_color=colors,
            text=['✅ PASS' if s else '❌ FAIL' for s in df_guardrails['Status']],
            textposition='inside'
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title="Guardrail Check",
            yaxis_title="Status",
            yaxis=dict(tickvals=[0, 1], ticktext=['FAIL', 'PASS']),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Risk factors
        st.markdown("### ⚠️ Risk Factors Detected")
        
        if risk.risk_factors:
            for i, factor in enumerate(risk.risk_factors, 1):
                st.warning(f"{i}. {factor}")
        else:
            st.success("✅ No major risk factors identified")
        
        st.markdown("---")
        
        # Confidence breakdown
        st.markdown("### 📊 Confidence Breakdown")
        
        # Mock confidence components (in real scenario from analysis result)
        conf_components = {
            'Sentiment': 0.80,
            'Technical': 0.75,
            'Fundamental': 0.68
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(conf_components.keys()),
                values=list(conf_components.values()),
                hole=0.4,
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
            )])
            
            fig.update_layout(
                height=300,
                title="Confidence Components",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart with weights
            weights = {
                'Sentiment': 0.30,
                'Technical': 0.30,
                'Fundamental': 0.40
            }
            
            weighted_conf = {k: conf_components[k] * weights[k] 
                           for k in conf_components.keys()}
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Raw Confidence',
                x=list(conf_components.keys()),
                y=list(conf_components.values()),
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Weighted',
                x=list(weighted_conf.keys()),
                y=list(weighted_conf.values()),
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                height=300,
                title="Weighted Confidence",
                yaxis=dict(range=[0, 1]),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Risk reasoning
        st.markdown("### 📝 Risk Assessment Reasoning")
        st.info(risk.reasoning)
        
        st.markdown("---")
        
        # Risk heatmap for portfolio
        st.markdown("### 🔥 Portfolio Risk Heatmap")
        
        if 'portfolio_analyses' in st.session_state and st.session_state.portfolio_analyses:
            # Create risk matrix
            portfolio_risk = []
            
            for sym, analysis in st.session_state.portfolio_analyses.items():
                risk_data = analysis.get('risk', {})
                portfolio_risk.append({
                    'Symbol': sym,
                    'Volatility': np.random.uniform(0.1, 0.4),  # Mock data
                    'Liquidity': np.random.uniform(0.5, 1.0),
                    'Confidence': analysis.get('confidence', 0),
                    'Risk Score': {'LOW': 0.25, 'MEDIUM': 0.5, 'HIGH': 0.75, 'EXTREME': 1.0}.get(
                        risk_data.get('risk_level', 'MEDIUM'), 0.5
                    )
                })
            
            df_risk = pd.DataFrame(portfolio_risk)
            
            # Heatmap
            fig = px.imshow(
                df_risk[['Volatility', 'Liquidity', 'Confidence', 'Risk Score']].T,
                labels=dict(x="Stock", y="Risk Factor", color="Score"),
                x=df_risk['Symbol'],
                y=['Volatility', 'Liquidity', 'Confidence', 'Risk Score'],
                color_continuous_scale='RdYlGn_r',
                aspect="auto"
            )
            
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Analyze portfolio stocks to see risk heatmap")
    
    else:
        # Placeholder
        st.info("👆 Enter a stock symbol and click **Analyze Risk** to get started")
        
        st.markdown("### 🛡️ Risk Management Framework")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Guardrail Checks")
            st.markdown("""
            - **Volatility Check**: Ensures 30-day volatility ≤ threshold
            - **Liquidity Check**: Validates sufficient daily volume
            - **Confidence Check**: Requires minimum confidence score
            - **Conflict Check**: Detects agent disagreements
            """)
        
        with col2:
            st.markdown("#### Risk Levels")
            st.markdown("""
            - 🟢 **LOW**: All guardrails pass, high confidence
            - 🟡 **MEDIUM**: Minor concerns, acceptable risk
            - 🟠 **HIGH**: Multiple warnings, proceed with caution
            - 🔴 **EXTREME**: Trade rejected, unacceptable risk
            """)
