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
    
    # NEW: Advanced Risk Metrics
    beta: Optional[float]  # Market sensitivity
    sharpe_ratio: Optional[float]  # Risk-adjusted return
    max_drawdown: Optional[float]  # Maximum historical drawdown
    value_at_risk: Optional[float]  # VaR (95% confidence)
    sortino_ratio: Optional[float]  # Downside risk-adjusted return
    
    # NEW: Position Risk
    position_risk: str  # "LOW", "MEDIUM", "HIGH"
    suggested_position_size: Optional[float]  # As % of portfolio
    stop_loss_price: Optional[float]  # Suggested stop-loss
    take_profit_price: Optional[float]  # Suggested take-profit
    risk_reward_ratio: Optional[float]  # Risk/Reward ratio
    
    # Final Decision
    trade_approved: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    
    # Reasoning
    reasoning: List[str]
    rejection_reasons: List[str]
    risk_factors: List[str]  # NEW: Detailed risk factors
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
        
        # NEW: Calculate advanced risk metrics
        beta = self._calculate_beta(symbol)
        sharpe_ratio = self._calculate_sharpe_ratio(symbol)
        max_drawdown = self._calculate_max_drawdown(symbol)
        value_at_risk = self._calculate_var(symbol, volatility_value)
        sortino_ratio = self._calculate_sortino_ratio(symbol)
        
        # NEW: Calculate position risk
        current_price = self._get_current_price(symbol)
        position_risk, suggested_size = self._calculate_position_risk(
            volatility_value, beta, max_drawdown
        )
        stop_loss, take_profit, risk_reward = self._calculate_trade_levels(
            current_price, volatility_value, technical
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(
            volatility_risk, liquidity_risk, conflict_risk, overall_confidence,
            beta, max_drawdown
        )
        
        # Final approval decision
        trade_approved = (
            passes_volatility and
            passes_liquidity and
            passes_confidence and
            passes_conflict
        )
        
        # Generate reasoning and risk factors
        reasoning, rejection_reasons = self._generate_reasoning(
            trade_approved,
            volatility_risk, liquidity_risk, conflict_risk,
            overall_confidence, passes_confidence,
            conflicts
        )
        
        # NEW: Generate detailed risk factors list
        risk_factors = self._generate_risk_factors(
            volatility_value, beta, max_drawdown, value_at_risk,
            liquidity_risk, conflict_risk
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
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            value_at_risk=value_at_risk,
            sortino_ratio=sortino_ratio,
            position_risk=position_risk,
            suggested_position_size=suggested_size,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            risk_reward_ratio=risk_reward,
            trade_approved=trade_approved,
            risk_level=risk_level,
            reasoning=reasoning,
            rejection_reasons=rejection_reasons,
            risk_factors=risk_factors,
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
            print(f"📊 Volatility check: {symbol} = {volatility:.2%}")
        except Exception as e:
            print(f"⚠️ Volatility check failed for {symbol}: {str(e)}")
            volatility = 0.25  # Moderate default if data unavailable
        
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
            print(f"📊 Liquidity check: {symbol} volume = {volume:,}")
        except Exception as e:
            print(f"⚠️ Liquidity check failed for {symbol}: {str(e)}")
            # Try alternative method - get historical data
            try:
                hist = self.market_data.get_historical_data(symbol, period="5d")
                if hist is not None and not hist.empty and 'Volume' in hist.columns:
                    volume = int(hist['Volume'].iloc[-1])
                    print(f"📊 Fallback volume from history: {volume:,}")
                else:
                    volume = 0
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
        agent_names = []
        
        if sentiment and hasattr(sentiment, 'confidence'):
            confidences.append(sentiment.confidence)
            weights.append(0.3)  # 30% weight
            agent_names.append("Sentiment")
        
        if technical and hasattr(technical, 'confidence'):
            confidences.append(technical.confidence)
            weights.append(0.3)  # 30% weight
            agent_names.append("Technical")
        
        if fundamental and hasattr(fundamental, 'confidence'):
            confidences.append(fundamental.confidence)
            weights.append(0.4)  # 40% weight (fundamental more important)
            agent_names.append("Fundamental")
        
        if not confidences:
            print("⚠️ No agent confidence scores available - defaulting to 0.5")
            return 0.5  # Return moderate confidence instead of 0
        
        # Log which agents contributed
        print(f"📊 Confidence from {len(confidences)} agents: {', '.join(agent_names)}")
        for name, conf in zip(agent_names, confidences):
            print(f"   - {name}: {conf:.1%}")
        
        # Weighted average (renormalize weights if some agents missing)
        total_weight = sum(weights)
        weighted_conf = sum(c * w for c, w in zip(confidences, weights))
        final_conf = weighted_conf / total_weight
        
        print(f"📊 Overall confidence: {final_conf:.1%}")
        return final_conf
    
    def _determine_risk_level(
        self,
        volatility_risk: str,
        liquidity_risk: str,
        conflict_risk: str,
        confidence: float,
        beta: Optional[float] = None,
        max_drawdown: Optional[float] = None
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
        
        # NEW: Adjust for beta
        if beta is not None:
            if abs(beta) > 1.5:
                total_risk += 1
            elif abs(beta) > 2.0:
                total_risk += 2
        
        # NEW: Adjust for max drawdown
        if max_drawdown is not None:
            if max_drawdown > 0.20:  # More than 20% drawdown
                total_risk += 1
            if max_drawdown > 0.30:  # More than 30% drawdown
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
    
    def _calculate_beta(self, symbol: str) -> Optional[float]:
        """Calculate stock beta relative to market (S&P 500)"""
        try:
            # Get stock and market data
            stock_data = self.market_data.get_historical_data(symbol, period="1y", interval="1d")
            market_data = self.market_data.get_historical_data("SPY", period="1y", interval="1d")
            
            if stock_data.empty or market_data.empty:
                return None
            
            # Calculate daily returns
            stock_returns = stock_data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align data
            min_len = min(len(stock_returns), len(market_returns))
            stock_returns = stock_returns.iloc[-min_len:]
            market_returns = market_returns.iloc[-min_len:]
            
            # Calculate beta using covariance/variance
            import numpy as np
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            if market_variance > 0:
                return covariance / market_variance
            return None
        except Exception:
            return None
    
    def _calculate_sharpe_ratio(self, symbol: str, risk_free_rate: float = 0.04) -> Optional[float]:
        """Calculate Sharpe ratio (risk-adjusted return)"""
        try:
            stock_data = self.market_data.get_historical_data(symbol, period="1y", interval="1d")
            if stock_data.empty:
                return None
            
            import numpy as np
            returns = stock_data['Close'].pct_change().dropna()
            
            # Annualized return and volatility
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            if annual_volatility > 0:
                return (annual_return - risk_free_rate) / annual_volatility
            return None
        except Exception:
            return None
    
    def _calculate_sortino_ratio(self, symbol: str, risk_free_rate: float = 0.04) -> Optional[float]:
        """Calculate Sortino ratio (downside risk-adjusted return)"""
        try:
            stock_data = self.market_data.get_historical_data(symbol, period="1y", interval="1d")
            if stock_data.empty:
                return None
            
            import numpy as np
            returns = stock_data['Close'].pct_change().dropna()
            
            # Annualized return
            annual_return = returns.mean() * 252
            
            # Downside deviation (only negative returns)
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252)
            
            if downside_deviation > 0:
                return (annual_return - risk_free_rate) / downside_deviation
            return None
        except Exception:
            return None
    
    def _calculate_max_drawdown(self, symbol: str) -> Optional[float]:
        """Calculate maximum drawdown from peak"""
        try:
            stock_data = self.market_data.get_historical_data(symbol, period="1y", interval="1d")
            if stock_data.empty:
                return None
            
            prices = stock_data['Close']
            peak = prices.expanding(min_periods=1).max()
            drawdown = (prices - peak) / peak
            
            return abs(drawdown.min())
        except Exception:
            return None
    
    def _calculate_var(self, symbol: str, volatility: float, confidence_level: float = 0.95) -> Optional[float]:
        """Calculate Value at Risk (VaR) at given confidence level"""
        try:
            import numpy as np
            # Using parametric VaR
            z_score = 1.645 if confidence_level == 0.95 else 2.326  # 95% or 99%
            daily_var = volatility * z_score
            return daily_var
        except Exception:
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        try:
            market_data = self.market_data.get_current_data(symbol)
            return market_data.price
        except Exception:
            return None
    
    def _calculate_position_risk(
        self,
        volatility: float,
        beta: Optional[float],
        max_drawdown: Optional[float]
    ) -> tuple:
        """Calculate position risk and suggested position size"""
        # Base position size (as % of portfolio)
        base_size = 5.0  # 5% default
        
        # Adjust for volatility
        if volatility > 0.30:
            base_size *= 0.5
        elif volatility > 0.20:
            base_size *= 0.75
        
        # Adjust for beta
        if beta is not None and abs(beta) > 1.5:
            base_size *= 0.75
        
        # Adjust for drawdown risk
        if max_drawdown is not None and max_drawdown > 0.25:
            base_size *= 0.75
        
        # Determine risk level
        if base_size >= 4.0:
            risk = "LOW"
        elif base_size >= 2.5:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        return risk, base_size
    
    def _calculate_trade_levels(
        self,
        current_price: Optional[float],
        volatility: float,
        technical: Optional[TechnicalSignals]
    ) -> tuple:
        """Calculate stop-loss, take-profit, and risk/reward ratio"""
        if current_price is None:
            return None, None, None
        
        # Use ATR-based stops if available from technical analysis
        if technical and hasattr(technical, 'atr'):
            atr = technical.atr
            stop_loss = current_price - (2 * atr)  # 2 ATR stop
            take_profit = current_price + (3 * atr)  # 3 ATR target
        else:
            # Use volatility-based stops
            stop_distance = current_price * volatility * 0.5  # 0.5x volatility
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * 2)  # 2:1 R/R
        
        # Calculate risk/reward ratio
        risk = current_price - stop_loss
        reward = take_profit - current_price
        risk_reward = reward / risk if risk > 0 else None
        
        return stop_loss, take_profit, risk_reward
    
    def _generate_risk_factors(
        self,
        volatility: float,
        beta: Optional[float],
        max_drawdown: Optional[float],
        var: Optional[float],
        liquidity_risk: str,
        conflict_risk: str
    ) -> List[str]:
        """Generate detailed list of risk factors"""
        factors = []
        
        # Volatility risk
        if volatility > 0.30:
            factors.append(f"HIGH volatility ({volatility:.1%}) - Expect large price swings")
        elif volatility > 0.20:
            factors.append(f"MODERATE volatility ({volatility:.1%}) - Normal market fluctuations")
        
        # Beta risk
        if beta is not None:
            if beta > 1.5:
                factors.append(f"HIGH beta ({beta:.2f}) - Amplified market moves")
            elif beta < 0.5:
                factors.append(f"LOW beta ({beta:.2f}) - Defensive stock")
            elif beta < 0:
                factors.append(f"NEGATIVE beta ({beta:.2f}) - Counter-market moves")
        
        # Drawdown risk
        if max_drawdown is not None and max_drawdown > 0.20:
            factors.append(f"SIGNIFICANT drawdown history ({max_drawdown:.1%} max)")
        
        # VaR risk
        if var is not None and var > 0.05:
            factors.append(f"Daily VaR of {var:.1%} at 95% confidence")
        
        # Liquidity risk
        if liquidity_risk == "HIGH":
            factors.append("LOW liquidity - May face slippage on large orders")
        
        # Signal conflict risk
        if conflict_risk == "HIGH":
            factors.append("CONFLICTING signals - Analysis uncertainty")
        
        return factors
    
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
