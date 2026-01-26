"""
Risk & Critic Agent
Validates signals from other agents and enforces safety guardrails
Acts as the final safety check before trade execution
"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
from agents.sentiment_agent import SentimentScore
from agents.technical_agent import TechnicalSignals
from agents.fundamental_agent import FundamentalScore
from data.market_data import MarketDataProvider


class RiskAssessment(BaseModel):
    """Structured risk assessment output"""
    symbol: str
    
    # Risk Factors
    volatility_risk: str  # "LOW", "MEDIUM", "HIGH"
    volatility_value: float
    
    liquidity_risk: str  # "LOW", "MEDIUM", "HIGH"
    volume: int
    
    signal_conflict_risk: str  # "LOW", "MEDIUM", "HIGH"
    conflicting_signals: List[str]
    
    # Confidence Analysis
    overall_confidence: float  # 0.0 to 1.0
    confidence_sufficient: bool  # >= threshold?
    
    # Guardrail Checks
    passes_volatility_check: bool
    passes_liquidity_check: bool
    passes_confidence_check: bool
    passes_conflict_check: bool
    
    # Final Decision
    trade_approved: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    
    # Reasoning
    reasoning: List[str]
    rejection_reasons: List[str]
    timestamp: datetime


class RiskAgent:
    """
    Risk validation and safety guardrail enforcement
    Final gate before trade execution
    """
    
    def __init__(
        self,
        min_confidence: float = 0.75,
        max_volatility: float = 0.30,
        min_volume: int = 1_000_000,
        max_conflict_tolerance: float = 0.3
    ):
        """
        Initialize risk agent with guardrail parameters
        
        Args:
            min_confidence: Minimum confidence threshold (default 0.75)
            max_volatility: Maximum acceptable volatility (default 30%)
            min_volume: Minimum daily volume for liquidity (default 1M)
            max_conflict_tolerance: Max % of conflicting signals (default 30%)
        """
        self.min_confidence = min_confidence
        self.max_volatility = max_volatility
        self.min_volume = min_volume
        self.max_conflict_tolerance = max_conflict_tolerance
        
        self.market_data = MarketDataProvider()
    
    def assess_risk(
        self,
        symbol: str,
        sentiment: Optional[SentimentScore] = None,
        technical: Optional[TechnicalSignals] = None,
        fundamental: Optional[FundamentalScore] = None
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment across all signals
        
        Args:
            symbol: Stock ticker
            sentiment: Sentiment analysis results
            technical: Technical analysis results
            fundamental: Fundamental analysis results
            
        Returns:
            RiskAssessment with approval decision
        """
        # Check volatility
        volatility_risk, volatility_value, passes_volatility = self._check_volatility(symbol)
        
        # Check liquidity
        liquidity_risk, volume, passes_liquidity = self._check_liquidity(symbol)
        
        # Check signal conflicts
        conflict_risk, conflicts, passes_conflict = self._check_signal_conflicts(
            sentiment, technical, fundamental
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            sentiment, technical, fundamental
        )
        passes_confidence = overall_confidence >= self.min_confidence
        
        # Determine risk level
        risk_level = self._determine_risk_level(
            volatility_risk, liquidity_risk, conflict_risk, overall_confidence
        )
        
        # Final approval decision
        trade_approved = (
            passes_volatility and
            passes_liquidity and
            passes_confidence and
            passes_conflict
        )
        
        # Generate reasoning
        reasoning, rejection_reasons = self._generate_reasoning(
            trade_approved,
            volatility_risk, liquidity_risk, conflict_risk,
            overall_confidence, passes_confidence,
            conflicts
        )
        
        return RiskAssessment(
            symbol=symbol,
            volatility_risk=volatility_risk,
            volatility_value=float(volatility_value),
            liquidity_risk=liquidity_risk,
            volume=int(volume),
            signal_conflict_risk=conflict_risk,
            conflicting_signals=conflicts,
            overall_confidence=float(overall_confidence),
            confidence_sufficient=passes_confidence,
            passes_volatility_check=passes_volatility,
            passes_liquidity_check=passes_liquidity,
            passes_confidence_check=passes_confidence,
            passes_conflict_check=passes_conflict,
            trade_approved=trade_approved,
            risk_level=risk_level,
            reasoning=reasoning,
            rejection_reasons=rejection_reasons,
            timestamp=datetime.now()
        )
    
    def _check_volatility(self, symbol: str) -> tuple:
        """
        Check if volatility is within acceptable range
        
        Returns:
            (risk_level, volatility_value, passes_check)
        """
        try:
            volatility = self.market_data.get_volatility(symbol, period="1mo")
        except:
            volatility = 0.5  # Conservative default if data unavailable
        
        passes = volatility <= self.max_volatility
        
        if volatility < 0.15:
            risk = "LOW"
        elif volatility < self.max_volatility:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        return risk, volatility, passes
    
    def _check_liquidity(self, symbol: str) -> tuple:
        """
        Check if stock has sufficient liquidity
        
        Returns:
            (risk_level, volume, passes_check)
        """
        try:
            market_data = self.market_data.get_current_data(symbol)
            volume = market_data.volume
        except:
            volume = 0
        
        passes = volume >= self.min_volume
        
        if volume > 10_000_000:
            risk = "LOW"
        elif volume >= self.min_volume:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        return risk, volume, passes
    
    def _check_signal_conflicts(
        self,
        sentiment: Optional[SentimentScore],
        technical: Optional[TechnicalSignals],
        fundamental: Optional[FundamentalScore]
    ) -> tuple:
        """
        Check for conflicting signals across agents
        
        Returns:
            (risk_level, conflict_list, passes_check)
        """
        conflicts = []
        signals = []
        
        # Collect all signals
        if sentiment:
            if sentiment.label == "positive":
                signals.append(("Sentiment", "BULLISH"))
            elif sentiment.label == "negative":
                signals.append(("Sentiment", "BEARISH"))
            else:
                signals.append(("Sentiment", "NEUTRAL"))
        
        if technical:
            signals.append(("Technical", technical.overall_trend))
        
        if fundamental:
            if fundamental.overall_signal in ["STRONG_BUY", "BUY"]:
                signals.append(("Fundamental", "BULLISH"))
            elif fundamental.overall_signal == "AVOID":
                signals.append(("Fundamental", "BEARISH"))
            else:
                signals.append(("Fundamental", "NEUTRAL"))
        
        # Count bullish/bearish signals
        bullish = sum(1 for _, s in signals if s == "BULLISH")
        bearish = sum(1 for _, s in signals if s == "BEARISH")
        neutral = sum(1 for _, s in signals if s == "NEUTRAL")
        total = len(signals)
        
        if total == 0:
            return "HIGH", ["No signals available"], False
        
        # Detect conflicts
        if bullish > 0 and bearish > 0:
            # We have conflicting signals
            for agent, signal in signals:
                if (signal == "BULLISH" and bearish > bullish) or \
                   (signal == "BEARISH" and bullish > bearish):
                    conflicts.append(f"{agent} signal conflicts with majority")
        
        # Calculate conflict ratio
        conflict_ratio = len(conflicts) / total if total > 0 else 1.0
        passes = conflict_ratio <= self.max_conflict_tolerance
        
        # Determine risk
        if conflict_ratio == 0:
            risk = "LOW"
        elif conflict_ratio <= self.max_conflict_tolerance:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        return risk, conflicts, passes
    
    def _calculate_overall_confidence(
        self,
        sentiment: Optional[SentimentScore],
        technical: Optional[TechnicalSignals],
        fundamental: Optional[FundamentalScore]
    ) -> float:
        """
        Calculate weighted average confidence across all signals
        
        Returns:
            Overall confidence (0.0 to 1.0)
        """
        confidences = []
        weights = []
        
        if sentiment:
            confidences.append(sentiment.confidence)
            weights.append(0.3)  # 30% weight
        
        if technical:
            confidences.append(technical.confidence)
            weights.append(0.3)  # 30% weight
        
        if fundamental:
            confidences.append(fundamental.confidence)
            weights.append(0.4)  # 40% weight (fundamental more important)
        
        if not confidences:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_conf = sum(c * w for c, w in zip(confidences, weights))
        
        return weighted_conf / total_weight
    
    def _determine_risk_level(
        self,
        volatility_risk: str,
        liquidity_risk: str,
        conflict_risk: str,
        confidence: float
    ) -> str:
        """Determine overall risk level"""
        risk_scores = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2
        }
        
        total_risk = (
            risk_scores[volatility_risk] +
            risk_scores[liquidity_risk] +
            risk_scores[conflict_risk]
        )
        
        # Adjust for confidence
        if confidence < 0.5:
            total_risk += 2
        elif confidence < 0.75:
            total_risk += 1
        
        # Map to risk level
        if total_risk == 0:
            return "LOW"
        elif total_risk <= 2:
            return "MEDIUM"
        elif total_risk <= 4:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _generate_reasoning(
        self,
        trade_approved: bool,
        volatility_risk: str,
        liquidity_risk: str,
        conflict_risk: str,
        overall_confidence: float,
        passes_confidence: bool,
        conflicts: List[str]
    ) -> tuple:
        """
        Generate reasoning and rejection reasons
        
        Returns:
            (reasoning_list, rejection_reasons_list)
        """
        reasoning = []
        rejection_reasons = []
        
        # Volatility
        if volatility_risk == "LOW":
            reasoning.append("✅ Volatility within acceptable range")
        elif volatility_risk == "MEDIUM":
            reasoning.append("⚠️ Moderate volatility - monitor closely")
        else:
            reasoning.append("❌ High volatility detected")
            rejection_reasons.append(f"Volatility risk too high ({volatility_risk})")
        
        # Liquidity
        if liquidity_risk == "LOW":
            reasoning.append("✅ Excellent liquidity")
        elif liquidity_risk == "MEDIUM":
            reasoning.append("⚠️ Adequate liquidity")
        else:
            reasoning.append("❌ Insufficient liquidity")
            rejection_reasons.append(f"Liquidity risk too high ({liquidity_risk})")
        
        # Confidence
        if passes_confidence:
            reasoning.append(f"✅ Confidence above threshold ({overall_confidence:.1%})")
        else:
            reasoning.append(f"❌ Confidence below threshold ({overall_confidence:.1%} < {self.min_confidence:.1%})")
            rejection_reasons.append(f"Confidence too low ({overall_confidence:.1%})")
        
        # Signal conflicts
        if conflict_risk == "LOW":
            reasoning.append("✅ All signals aligned")
        elif conflict_risk == "MEDIUM":
            reasoning.append("⚠️ Minor signal disagreement")
        else:
            reasoning.append("❌ Major signal conflicts detected")
            rejection_reasons.append("Conflicting signals from multiple agents")
            for conflict in conflicts:
                rejection_reasons.append(f"  • {conflict}")
        
        # Final verdict
        if trade_approved:
            reasoning.append("✅ TRADE APPROVED - All guardrails passed")
        else:
            reasoning.append("❌ TRADE REJECTED - Failed safety checks")
        
        return reasoning, rejection_reasons


# Example usage
if __name__ == "__main__":
    from agents.sentiment_agent import NewsSentimentAgent
    from agents.technical_agent import TechnicalAnalysisAgent
    from agents.fundamental_agent import FundamentalAnalysisAgent
    
    symbol = "AAPL"
    
    print(f"\n{'='*60}")
    print(f"Risk Assessment: {symbol}")
    print(f"{'='*60}\n")
    
    try:
        # Run all agents
        print("Running sentiment analysis...")
        sentiment_agent = NewsSentimentAgent()
        market_data = MarketDataProvider()
        news = market_data.get_news(symbol)
        sentiment = sentiment_agent.analyze_news_batch(news)
        
        print("Running technical analysis...")
        ta_agent = TechnicalAnalysisAgent()
        technical = ta_agent.analyze(symbol)
        
        print("Running fundamental analysis...")
        fa_agent = FundamentalAnalysisAgent()
        fundamental = fa_agent.analyze(symbol)
        
        print("Performing risk assessment...\n")
        risk_agent = RiskAgent()
        risk = risk_agent.assess_risk(
            symbol=symbol,
            sentiment=sentiment,
            technical=technical,
            fundamental=fundamental
        )
        
        # Display results
        print(f"🎯 Overall Confidence: {risk.overall_confidence:.1%}")
        print(f"📊 Risk Level: {risk.risk_level}")
        print(f"\n🔍 Risk Factors:")
        print(f"  • Volatility: {risk.volatility_risk} ({risk.volatility_value:.1%})")
        print(f"  • Liquidity: {risk.liquidity_risk} ({risk.volume:,} shares)")
        print(f"  • Signal Conflicts: {risk.signal_conflict_risk}")
        
        print(f"\n✓ Guardrail Checks:")
        print(f"  • Volatility: {'PASS' if risk.passes_volatility_check else 'FAIL'}")
        print(f"  • Liquidity: {'PASS' if risk.passes_liquidity_check else 'FAIL'}")
        print(f"  • Confidence: {'PASS' if risk.passes_confidence_check else 'FAIL'}")
        print(f"  • Conflicts: {'PASS' if risk.passes_conflict_check else 'FAIL'}")
        
        print(f"\n💡 Analysis:")
        for reason in risk.reasoning:
            print(f"  {reason}")
        
        if not risk.trade_approved:
            print(f"\n❌ Rejection Reasons:")
            for reason in risk.rejection_reasons:
                print(f"  {reason}")
        
        print(f"\n{'='*60}")
        print(f"FINAL DECISION: {'✅ APPROVED' if risk.trade_approved else '❌ REJECTED'}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
