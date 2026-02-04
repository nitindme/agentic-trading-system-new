"""
LangGraph Trading Workflow
Orchestrates multi-agent analysis with conditional routing
"""

from typing import Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, END
from graph.state import TradingState
from agents.sentiment_agent import NewsSentimentAgent, EnsembleSentimentAgent
from agents.technical_agent import TechnicalAnalysisAgent
from agents.fundamental_agent import FundamentalAnalysisAgent
from agents.risk_agent import RiskAgent
from data.market_data import MarketDataProvider


class TradingWorkflow:
    """
    LangGraph workflow for multi-agent trading analysis
    
    Flow:
    1. Fetch Data
    2. Sentiment Analysis
    3. Technical Analysis
    4. Fundamental Analysis
    5. Risk Assessment
    6. Decision (conditional routing)
       - If approved → Execute
       - If rejected → Explain rejection
    """
    
    def __init__(
        self,
        min_confidence: float = 0.75,
        max_volatility: float = 0.30
    ):
        """
        Initialize trading workflow
        
        Args:
            min_confidence: Minimum confidence for trade approval
            max_volatility: Maximum acceptable volatility
        """
        # Initialize agents
        self.sentiment_agent = NewsSentimentAgent()
        self.technical_agent = TechnicalAnalysisAgent()
        self.fundamental_agent = FundamentalAnalysisAgent()
        self.risk_agent = RiskAgent(
            min_confidence=min_confidence,
            max_volatility=max_volatility
        )
        self.market_data = MarketDataProvider()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Create graph
        workflow = StateGraph(TradingState)
        
        # Add nodes (agents)
        workflow.add_node("fetch_data", self._fetch_data)
        workflow.add_node("sentiment_analysis", self._analyze_sentiment)
        workflow.add_node("technical_analysis", self._analyze_technical)
        workflow.add_node("fundamental_analysis", self._analyze_fundamental)
        workflow.add_node("risk_assessment", self._assess_risk)
        workflow.add_node("make_decision", self._make_decision)
        workflow.add_node("explain_rejection", self._explain_rejection)
        
        # Define edges (flow)
        workflow.set_entry_point("fetch_data")
        workflow.add_edge("fetch_data", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "technical_analysis")
        workflow.add_edge("technical_analysis", "fundamental_analysis")
        workflow.add_edge("fundamental_analysis", "risk_assessment")
        
        # Conditional routing after risk assessment
        workflow.add_conditional_edges(
            "risk_assessment",
            self._should_proceed,
            {
                "approved": "make_decision",
                "rejected": "explain_rejection"
            }
        )
        
        # Both paths end the workflow
        workflow.add_edge("make_decision", END)
        workflow.add_edge("explain_rejection", END)
        
        return workflow.compile()
    
    # Node functions (each agent's logic)
    
    def _fetch_data(self, state: TradingState) -> Dict[str, Any]:
        """Fetch market data and news"""
        print(f"📡 Fetching data for {state['symbol']}...")
        
        try:
            # This node just validates the symbol
            # Actual data fetching happens in each agent
            market_data = self.market_data.get_current_data(state['symbol'])
            
            return {
                "timestamp": datetime.now(),
                "errors": []
            }
        except Exception as e:
            return {
                "errors": [f"Data fetch error: {str(e)}"],
                "trade_approved": False
            }
    
    def _analyze_sentiment(self, state: TradingState) -> Dict[str, Any]:
        """Run sentiment analysis"""
        print(f"😊 Analyzing sentiment for {state['symbol']}...")
        
        try:
            news = self.market_data.get_news(state['symbol'])
            sentiment = self.sentiment_agent.analyze_news_batch(news)
            
            return {
                "sentiment": sentiment
            }
        except Exception as e:
            errors = state.get("errors", [])
            errors.append(f"Sentiment analysis error: {str(e)}")
            print(f"⚠️ Sentiment analysis failed: {str(e)}")
            return {
                "sentiment": None,
                "errors": errors
            }
    
    def _analyze_technical(self, state: TradingState) -> Dict[str, Any]:
        """Run technical analysis"""
        print(f"📈 Analyzing technicals for {state['symbol']}...")
        
        try:
            technical = self.technical_agent.analyze(state['symbol'])
            
            return {
                "technical": technical
            }
        except Exception as e:
            errors = state.get("errors", [])
            errors.append(f"Technical analysis error: {str(e)}")
            print(f"⚠️ Technical analysis failed: {str(e)}")
            return {
                "technical": None,
                "errors": errors
            }
    
    def _analyze_fundamental(self, state: TradingState) -> Dict[str, Any]:
        """Run fundamental analysis"""
        print(f"💰 Analyzing fundamentals for {state['symbol']}...")
        
        try:
            fundamental = self.fundamental_agent.analyze(state['symbol'])
            
            return {
                "fundamental": fundamental
            }
        except Exception as e:
            errors = state.get("errors", [])
            errors.append(f"Fundamental analysis error: {str(e)}")
            print(f"⚠️ Fundamental analysis failed: {str(e)}")
            return {
                "fundamental": None,
                "errors": errors
            }
    
    def _assess_risk(self, state: TradingState) -> Dict[str, Any]:
        """Run risk assessment"""
        print(f"🛡️ Assessing risk for {state['symbol']}...")
        
        try:
            risk = self.risk_agent.assess_risk(
                symbol=state['symbol'],
                sentiment=state.get('sentiment'),
                technical=state.get('technical'),
                fundamental=state.get('fundamental')
            )
            
            return {
                "risk": risk,
                "trade_approved": risk.trade_approved,
                "confidence": risk.overall_confidence
            }
        except Exception as e:
            errors = state.get("errors", [])
            errors.append(f"Risk assessment error: {str(e)}")
            return {
                "errors": errors,
                "trade_approved": False
            }
    
    def _should_proceed(self, state: TradingState) -> str:
        """Conditional routing: approved or rejected?"""
        if state.get("trade_approved", False):
            return "approved"
        else:
            return "rejected"
    
    def _make_decision(self, state: TradingState) -> Dict[str, Any]:
        """Generate final trading recommendation"""
        print(f"🎯 Making trading decision for {state['symbol']}...")
        
        sentiment = state.get('sentiment')
        technical = state.get('technical')
        fundamental = state.get('fundamental')
        risk = state.get('risk')
        
        # Determine recommendation based on signals
        bullish_signals = 0
        bearish_signals = 0
        
        if sentiment and sentiment.label == "positive":
            bullish_signals += 1
        elif sentiment and sentiment.label == "negative":
            bearish_signals += 1
        
        if technical and technical.overall_trend == "BULLISH":
            bullish_signals += 1
        elif technical and technical.overall_trend == "BEARISH":
            bearish_signals += 1
        
        if fundamental and fundamental.overall_signal in ["STRONG_BUY", "BUY"]:
            bullish_signals += 1
        elif fundamental and fundamental.overall_signal == "AVOID":
            bearish_signals += 1
        
        # Make recommendation
        if bullish_signals > bearish_signals:
            recommendation = "BUY"
        elif bearish_signals > bullish_signals:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        # Compile reasoning
        reasoning = []
        if sentiment:
            reasoning.extend(sentiment.reasoning[:2])
        if technical:
            reasoning.extend(technical.reasoning[:2])
        if fundamental:
            reasoning.extend(fundamental.reasoning[:2])
        if risk:
            reasoning.extend(risk.reasoning[:3])
        
        return {
            "recommendation": recommendation,
            "reasoning": reasoning,
            "execution_status": "APPROVED_FOR_EXECUTION"
        }
    
    def _explain_rejection(self, state: TradingState) -> Dict[str, Any]:
        """Explain why trade was rejected"""
        print(f"❌ Trade rejected for {state['symbol']}")
        
        risk = state.get('risk')
        
        reasoning = [
            "Trade rejected by risk agent",
            f"Overall confidence: {risk.overall_confidence:.1%}" if risk else "Insufficient data"
        ]
        
        if risk and risk.rejection_reasons:
            reasoning.extend(risk.rejection_reasons)
        
        return {
            "recommendation": "HOLD",
            "reasoning": reasoning,
            "execution_status": "REJECTED"
        }
    
    # Public API
    
    def analyze_stock(
        self,
        symbol: str,
        mode: str = "SIMULATION"
    ) -> Dict[str, Any]:
        """
        Run complete multi-agent analysis
        
        Args:
            symbol: Stock ticker
            mode: Execution mode (SIMULATION, APPROVAL_REQUIRED, AUTO_EXECUTE)
            
        Returns:
            Complete analysis results
        """
        # Initialize state
        initial_state: TradingState = {
            "symbol": symbol,
            "mode": mode,
            "sentiment": None,
            "technical": None,
            "fundamental": None,
            "risk": None,
            "recommendation": None,
            "confidence": None,
            "reasoning": [],
            "trade_approved": False,
            "execution_status": None,
            "order_id": None,
            "timestamp": datetime.now(),
            "errors": []
        }
        
        # Run workflow
        print(f"\n{'='*60}")
        print(f"🤖 Starting Multi-Agent Analysis: {symbol}")
        print(f"{'='*60}\n")
        
        final_state = self.workflow.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"✅ Analysis Complete")
        print(f"{'='*60}\n")
        
        # Format output
        return self._format_output(final_state)
    
    def _format_output(self, state: TradingState) -> Dict[str, Any]:
        """Format final output for display"""
        
        sentiment = state.get('sentiment')
        technical = state.get('technical')
        fundamental = state.get('fundamental')
        risk = state.get('risk')
        
        return {
            "symbol": state['symbol'],
            "recommendation": state.get('recommendation'),
            "confidence": state.get('confidence', 0.0),
            "risk_level": risk.risk_level if risk else "UNKNOWN",
            "trade_approved": state.get('trade_approved', False),
            "execution_status": state.get('execution_status'),
            
            "sentiment": {
                "score": sentiment.score if sentiment else 0.0,
                "label": sentiment.label if sentiment else "N/A",
                "confidence": sentiment.confidence if sentiment else 0.0,
                "reasoning": sentiment.reasoning if sentiment else [],
                "sources": sentiment.sources if sentiment else []
            } if sentiment else None,
            
            "technical": {
                "overall_trend": technical.overall_trend if technical else "NEUTRAL",
                "trend_strength": technical.trend_strength if technical else 0.0,
                "rsi": technical.rsi if technical else 50.0,
                "rsi_signal": technical.rsi_signal if technical else "neutral",
                "macd": technical.macd if technical else 0.0,
                "macd_signal": technical.macd_signal if technical else 0.0,
                "macd_trend": technical.macd_trend if technical else "neutral",
                "sma_50": technical.sma_50 if technical else 0.0,
                "sma_200": technical.sma_200 if technical else 0.0,
                "ma_signal": technical.ma_signal if technical else "neutral",
                "price": technical.price if technical else 0.0,
                "volume": technical.volume if technical else 0,
                "volume_sma_20": technical.volume_sma_20 if technical else 0,
                "volume_trend": technical.volume_trend if technical else "stable",
                "bb_upper": technical.bb_upper if technical else 0.0,
                "bb_middle": technical.bb_middle if technical else 0.0,
                "bb_lower": technical.bb_lower if technical else 0.0,
                "bb_signal": technical.bb_signal if technical else "neutral",
                "reasoning": technical.reasoning if technical else "No reasoning available",
                "confidence": technical.confidence if technical else 0.0
            } if technical else None,
            
            "fundamental": {
                "overall_signal": fundamental.overall_signal if fundamental else "HOLD",
                "overall_score": fundamental.overall_score if fundamental else 0.0,
                "valuation_score": fundamental.valuation_score if fundamental else 0.0,
                "valuation_signal": fundamental.valuation_signal if fundamental else "FAIR",
                "growth_score": fundamental.growth_score if fundamental else 0.0,
                "growth_signal": fundamental.growth_signal if fundamental else "STABLE",
                "profitability_score": fundamental.profitability_score if fundamental else 0.0,
                "profitability_signal": fundamental.profitability_signal if fundamental else "MODERATE",
                "health_score": fundamental.health_score if fundamental else 0.0,
                "health_signal": fundamental.health_signal if fundamental else "STABLE",
                "pe_ratio": fundamental.pe_ratio if fundamental else None,
                "revenue_growth": fundamental.revenue_growth if fundamental else None,
                "profit_margin": fundamental.profit_margin if fundamental else None,
                "debt_to_equity": fundamental.debt_to_equity if fundamental else None,
                "reasoning": fundamental.reasoning if fundamental else "No reasoning available",
                "confidence": fundamental.confidence if fundamental else 0.0
            } if fundamental else None,
            
            "risk": {
                "risk_level": risk.risk_level if risk else "UNKNOWN",
                "trade_approved": risk.trade_approved if risk else False,
                "overall_confidence": risk.overall_confidence if risk else 0.0,
                "guardrails_passed": {
                    "volatility_check": risk.passes_volatility_check if risk else False,
                    "liquidity_check": risk.passes_liquidity_check if risk else False,
                    "confidence_check": risk.passes_confidence_check if risk else False,
                    "conflict_check": risk.passes_conflict_check if risk else False
                },
                "risk_factors": risk.rejection_reasons if risk else [],
                "reasoning": risk.reasoning if risk else []
            } if risk else None,
            
            "reasoning": state.get('reasoning', []),
            "errors": state.get('errors', []),
            "timestamp": state['timestamp']
        }


# Example usage
if __name__ == "__main__":
    import json
    
    # Initialize workflow
    workflow = TradingWorkflow(
        min_confidence=0.75,
        max_volatility=0.30
    )
    
    # Analyze a stock
    symbol = "AAPL"
    result = workflow.analyze_stock(symbol, mode="SIMULATION")
    
    # Display results
    print("\n" + "="*60)
    print("📊 FINAL RECOMMENDATION")
    print("="*60 + "\n")
    
    print(json.dumps(result, indent=2, default=str))
    
    print(f"\n🎯 Recommendation: {result['recommendation']}")
    print(f"📈 Confidence: {result['confidence']:.1%}")
    print(f"🛡️ Risk Level: {result['risk_level']}")
    print(f"✅ Trade Approved: {result['trade_approved']}")
    
    if result['reasoning']:
        print(f"\n💡 Key Insights:")
        for reason in result['reasoning'][:5]:
            print(f"  • {reason}")
