"""
Portfolio Tracking Page
Multi-stock comparison and watchlist management
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from graph.workflow import TradingWorkflow
from data.market_data import MarketDataProvider

def render():
    st.markdown("## 💼 Portfolio Dashboard")
    st.markdown("### Track and compare multiple stocks")
    st.markdown("---")
    
    # Initialize session state for watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    if 'portfolio_analyses' not in st.session_state:
        st.session_state.portfolio_analyses = {}
    
    # Watchlist management
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_symbol = st.text_input(
            "Add Stock to Watchlist",
            placeholder="Enter symbol (e.g., NVDA)",
            help="Add a new stock to your watchlist"
        ).upper()
    
    with col2:
        if st.button("➕ Add", use_container_width=True):
            if new_symbol and new_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol)
                st.success(f"✅ Added {new_symbol}")
                st.rerun()
            elif new_symbol in st.session_state.watchlist:
                st.warning(f"⚠️ {new_symbol} already in watchlist")
    
    with col3:
        if st.button("🔄 Analyze All", type="primary", use_container_width=True):
            with st.spinner("Analyzing all stocks..."):
                workflow = TradingWorkflow()
                
                for symbol in st.session_state.watchlist:
                    try:
                        result = workflow.analyze_stock(symbol, mode="SIMULATION")
                        st.session_state.portfolio_analyses[symbol] = result
                    except Exception as e:
                        st.error(f"Failed to analyze {symbol}: {str(e)}")
                
                st.success("✅ Analysis complete!")
                st.rerun()
    
    st.markdown("---")
    
    # Display watchlist
    st.markdown("### 📋 Your Watchlist")
    
    if st.session_state.watchlist:
        # Create watchlist table
        watchlist_data = []
        
        for symbol in st.session_state.watchlist:
            try:
                # Get market data
                market_data = MarketDataProvider()
                current_data = market_data.get_current_data(symbol)
                
                price = current_data.get('price', 0)
                change = current_data.get('change_percent', 0)
                volume = current_data.get('volume', 0)
                
                # Get analysis if available
                analysis = st.session_state.portfolio_analyses.get(symbol, {})
                recommendation = analysis.get('recommendation', '-')
                confidence = analysis.get('confidence', 0)
                trade_approved = analysis.get('trade_approved', False)
                
                watchlist_data.append({
                    'Symbol': symbol,
                    'Price': f"${price:.2f}",
                    'Change': f"{change:+.2f}%",
                    'Volume': f"{volume:,.0f}",
                    'Recommendation': recommendation,
                    'Confidence': f"{confidence:.1%}",
                    'Status': '✅' if trade_approved else '❌'
                })
            
            except Exception as e:
                watchlist_data.append({
                    'Symbol': symbol,
                    'Price': 'Error',
                    'Change': '-',
                    'Volume': '-',
                    'Recommendation': '-',
                    'Confidence': '-',
                    'Status': '❌'
                })
        
        # Display as dataframe
        df = pd.DataFrame(watchlist_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Change": st.column_config.TextColumn("Change %", width="small"),
                "Volume": st.column_config.TextColumn("Volume", width="medium"),
                "Recommendation": st.column_config.TextColumn("Rec", width="small"),
                "Confidence": st.column_config.TextColumn("Conf", width="small"),
                "Status": st.column_config.TextColumn("✓", width="small")
            }
        )
        
        # Remove stock section
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            remove_symbol = st.selectbox(
                "Remove from watchlist",
                st.session_state.watchlist,
                label_visibility="collapsed"
            )
        
        with col2:
            if st.button("🗑️ Remove", use_container_width=True):
                if remove_symbol in st.session_state.watchlist:
                    st.session_state.watchlist.remove(remove_symbol)
                    if remove_symbol in st.session_state.portfolio_analyses:
                        del st.session_state.portfolio_analyses[remove_symbol]
                    st.success(f"Removed {remove_symbol}")
                    st.rerun()
    
    else:
        st.info("👆 Add stocks to your watchlist to get started")
    
    st.markdown("---")
    
    # Portfolio analytics (if analyses available)
    if st.session_state.portfolio_analyses:
        st.markdown("### 📊 Portfolio Analytics")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_stocks = len(st.session_state.portfolio_analyses)
        buy_signals = sum(1 for a in st.session_state.portfolio_analyses.values() 
                         if a.get('recommendation') == 'BUY')
        approved_trades = sum(1 for a in st.session_state.portfolio_analyses.values() 
                            if a.get('trade_approved', False))
        avg_confidence = sum(a.get('confidence', 0) 
                           for a in st.session_state.portfolio_analyses.values()) / total_stocks
        
        with col1:
            st.metric("Total Analyzed", total_stocks)
        
        with col2:
            st.metric("Buy Signals", buy_signals, f"{buy_signals/total_stocks:.0%}")
        
        with col3:
            st.metric("Approved Trades", approved_trades, f"{approved_trades/total_stocks:.0%}")
        
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        st.markdown("---")
        
        # Recommendation distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Recommendation Distribution")
            
            rec_counts = {}
            for analysis in st.session_state.portfolio_analyses.values():
                rec = analysis.get('recommendation', 'UNKNOWN')
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            
            if rec_counts:
                fig = go.Figure(data=[go.Pie(
                    labels=list(rec_counts.keys()),
                    values=list(rec_counts.values()),
                    hole=0.4,
                    marker=dict(colors=['#28a745', '#ffc107', '#dc3545', '#6c757d'])
                )])
                
                fig.update_layout(
                    height=300,
                    showlegend=True,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🛡️ Risk Distribution")
            
            risk_counts = {}
            for analysis in st.session_state.portfolio_analyses.values():
                risk_level = analysis.get('risk', {}).get('risk_level', 'UNKNOWN')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            if risk_counts:
                fig = go.Figure(data=[go.Bar(
                    x=list(risk_counts.keys()),
                    y=list(risk_counts.values()),
                    marker=dict(color=['#28a745', '#ffc107', '#ff8c00', '#dc3545'])
                )])
                
                fig.update_layout(
                    height=300,
                    xaxis_title="Risk Level",
                    yaxis_title="Count",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Confidence comparison
        st.markdown("### 📊 Confidence Comparison")
        
        conf_data = []
        for symbol, analysis in st.session_state.portfolio_analyses.items():
            conf_data.append({
                'Symbol': symbol,
                'Overall': analysis.get('confidence', 0),
                'Sentiment': analysis.get('sentiment', {}).get('confidence', 0),
                'Technical': analysis.get('technical', {}).get('confidence', 0),
                'Fundamental': analysis.get('fundamental', {}).get('overall_score', 0)
            })
        
        df_conf = pd.DataFrame(conf_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Overall',
            x=df_conf['Symbol'],
            y=df_conf['Overall'],
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.add_trace(go.Bar(
            name='Sentiment',
            x=df_conf['Symbol'],
            y=df_conf['Sentiment'],
            marker_color='rgb(26, 118, 255)'
        ))
        
        fig.add_trace(go.Bar(
            name='Technical',
            x=df_conf['Symbol'],
            y=df_conf['Technical'],
            marker_color='rgb(50, 171, 96)'
        ))
        
        fig.add_trace(go.Bar(
            name='Fundamental',
            x=df_conf['Symbol'],
            y=df_conf['Fundamental'],
            marker_color='rgb(255, 133, 27)'
        ))
        
        fig.update_layout(
            barmode='group',
            height=400,
            xaxis_title="Stock",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison")
        
        comparison_data = []
        for symbol, analysis in st.session_state.portfolio_analyses.items():
            comparison_data.append({
                'Symbol': symbol,
                'Recommendation': analysis.get('recommendation', '-'),
                'Confidence': f"{analysis.get('confidence', 0):.1%}",
                'Risk': analysis.get('risk', {}).get('risk_level', '-'),
                'Sentiment': analysis.get('sentiment', {}).get('label', '-').upper(),
                'Technical': analysis.get('technical', {}).get('trend', '-'),
                'Fundamental': analysis.get('fundamental', {}).get('overall_signal', '-'),
                'Approved': '✅' if analysis.get('trade_approved', False) else '❌'
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        st.dataframe(
            df_comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Recommendation": st.column_config.TextColumn("Rec", width="small"),
                "Confidence": st.column_config.TextColumn("Conf", width="small"),
                "Risk": st.column_config.TextColumn("Risk", width="small"),
                "Sentiment": st.column_config.TextColumn("Sent", width="small"),
                "Technical": st.column_config.TextColumn("Tech", width="small"),
                "Fundamental": st.column_config.TextColumn("Fund", width="small"),
                "Approved": st.column_config.TextColumn("✓", width="small")
            }
        )
    
    else:
        st.info("👆 Click **Analyze All** to generate portfolio insights")
    
    st.markdown("---")
    
    # Export functionality
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 💾 Export Portfolio")
    
    with col2:
        if st.button("📄 Export CSV", use_container_width=True):
            if st.session_state.portfolio_analyses:
                # Create export data
                export_data = []
                for symbol, analysis in st.session_state.portfolio_analyses.items():
                    export_data.append({
                        'Symbol': symbol,
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Recommendation': analysis.get('recommendation', '-'),
                        'Confidence': analysis.get('confidence', 0),
                        'Risk_Level': analysis.get('risk', {}).get('risk_level', '-'),
                        'Trade_Approved': analysis.get('trade_approved', False),
                        'Sentiment_Score': analysis.get('sentiment', {}).get('score', 0),
                        'Technical_Trend': analysis.get('technical', {}).get('trend', '-'),
                        'Fundamental_Score': analysis.get('fundamental', {}).get('overall_score', 0)
                    })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
    
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.portfolio_analyses = {}
            st.success("Cleared all analyses")
            st.rerun()
