"""
Fundamental Analysis Agent
Evaluates company fundamentals: valuation, growth, profitability, financial health
"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
from data.market_data import MarketDataProvider


class FundamentalScore(BaseModel):
    """Structured fundamental analysis output"""
    symbol: str
    
    # Valuation Metrics
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    ev_to_ebitda: Optional[float]  # NEW: Enterprise Value to EBITDA
    price_to_sales: Optional[float]  # NEW: Price to Sales ratio
    valuation_score: float  # 0.0 to 1.0
    valuation_signal: str  # "UNDERVALUED", "FAIR", "OVERVALUED"
    
    # Growth Metrics
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    free_cash_flow_growth: Optional[float]  # NEW: FCF growth
    growth_score: float  # 0.0 to 1.0
    growth_signal: str  # "HIGH_GROWTH", "MODERATE", "LOW_GROWTH"
    
    # Profitability Metrics
    profit_margin: Optional[float]
    operating_margin: Optional[float]
    roe: Optional[float]  # Return on Equity
    roa: Optional[float]  # NEW: Return on Assets
    roic: Optional[float]  # NEW: Return on Invested Capital
    gross_margin: Optional[float]  # NEW: Gross Margin
    profitability_score: float  # 0.0 to 1.0
    profitability_signal: str  # "HIGHLY_PROFITABLE", "PROFITABLE", "STRUGGLING"
    
    # Financial Health
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    interest_coverage: Optional[float]  # NEW: Interest Coverage Ratio
    free_cash_flow: Optional[float]  # NEW: Free Cash Flow
    health_score: float  # 0.0 to 1.0
    health_signal: str  # "STRONG", "STABLE", "WEAK"
    
    # NEW: Dividend & Shareholder Returns
    dividend_yield: Optional[float]
    payout_ratio: Optional[float]
    buyback_yield: Optional[float]
    shareholder_yield: Optional[float]  # Dividend + Buyback yield
    
    # NEW: Efficiency Metrics
    asset_turnover: Optional[float]
    inventory_turnover: Optional[float]
    receivables_turnover: Optional[float]
    
    # Overall Assessment
    overall_score: float  # 0.0 to 1.0
    overall_signal: str  # "STRONG_BUY", "BUY", "HOLD", "AVOID"
    
    # Reasoning
    reasoning: List[str]
    confidence: float
    timestamp: datetime


class FundamentalAnalysisAgent:
    """
    Analyzes company fundamentals
    Evaluates financial health, growth potential, and valuation
    """
    
    def __init__(self):
        """Initialize fundamental analysis agent"""
        self.market_data = MarketDataProvider()
        
        # Benchmark thresholds (industry averages)
        self.GOOD_PE = 25  # P/E ratio below this is good
        self.GOOD_PEG = 1.5  # PEG below this is good
        self.GOOD_EV_EBITDA = 15  # EV/EBITDA below this is good
        self.GOOD_REVENUE_GROWTH = 0.10  # 10% YoY
        self.GOOD_PROFIT_MARGIN = 0.15  # 15%
        self.GOOD_ROE = 0.15  # 15%
        self.GOOD_ROA = 0.10  # 10%
        self.GOOD_ROIC = 0.12  # 12%
        self.GOOD_DEBT_TO_EQUITY = 0.5  # 0.5x
        self.GOOD_CURRENT_RATIO = 1.5  # 1.5x
        self.GOOD_INTEREST_COVERAGE = 5  # 5x interest coverage
    
    def analyze(self, symbol: str) -> FundamentalScore:
        """
        Perform comprehensive fundamental analysis
        
        Args:
            symbol: Stock ticker
            
        Returns:
            FundamentalScore with all metrics
        """
        # Fetch fundamental data
        try:
            fundamentals = self.market_data.get_fundamentals(symbol)
            print(f"📊 Fundamentals fetched for {symbol}: {len(fundamentals)} metrics")
            
            # Log key metrics
            pe = fundamentals.get('pe_ratio')
            div = fundamentals.get('dividend_yield')
            fcf = fundamentals.get('free_cash_flow')
            print(f"   - P/E: {pe}, Dividend: {div}, FCF: {fcf}")
        except Exception as e:
            print(f"⚠️ Error fetching fundamentals for {symbol}: {str(e)}")
            fundamentals = {}
        
        if not fundamentals:
            print(f"⚠️ No fundamental data available for: {symbol}")
            # Return default values instead of raising error
            return FundamentalScore(
                symbol=symbol,
                pe_ratio=None, forward_pe=None, peg_ratio=None, price_to_book=None,
                ev_to_ebitda=None, price_to_sales=None,
                valuation_score=0.5, valuation_signal="FAIR",
                revenue_growth=None, earnings_growth=None, free_cash_flow_growth=None,
                growth_score=0.5, growth_signal="MODERATE",
                profit_margin=None, operating_margin=None, roe=None, roa=None, roic=None, gross_margin=None,
                profitability_score=0.5, profitability_signal="MODERATE",
                debt_to_equity=None, current_ratio=None, quick_ratio=None, interest_coverage=None, free_cash_flow=None,
                health_score=0.5, health_signal="STABLE",
                dividend_yield=None, payout_ratio=None, buyback_yield=None, shareholder_yield=None,
                asset_turnover=None, inventory_turnover=None, receivables_turnover=None,
                overall_score=0.5, overall_signal="HOLD",
                reasoning=["Fundamental data unavailable - using neutral defaults"],
                confidence=0.3,
                timestamp=datetime.now()
            )
        
        # Analyze each category
        valuation_score, valuation_signal = self._analyze_valuation(fundamentals)
        growth_score, growth_signal = self._analyze_growth(fundamentals)
        profitability_score, profitability_signal = self._analyze_profitability(fundamentals)
        health_score, health_signal = self._analyze_financial_health(fundamentals)
        
        # Calculate overall score (weighted average)
        overall_score = (
            valuation_score * 0.3 +
            growth_score * 0.3 +
            profitability_score * 0.2 +
            health_score * 0.2
        )
        
        # Determine overall signal
        overall_signal = self._determine_overall_signal(overall_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            fundamentals,
            valuation_signal,
            growth_signal,
            profitability_signal,
            health_signal
        )
        
        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(fundamentals)
        
        # Extract shareholder return metrics
        dividend_yield = fundamentals.get('dividend_yield')
        payout_ratio = fundamentals.get('payout_ratio')
        buyback_yield = fundamentals.get('buyback_yield', 0)
        shareholder_yield = (dividend_yield or 0) + (buyback_yield or 0)
        
        return FundamentalScore(
            symbol=symbol,
            pe_ratio=fundamentals.get('pe_ratio'),
            forward_pe=fundamentals.get('forward_pe'),
            peg_ratio=fundamentals.get('peg_ratio'),
            price_to_book=fundamentals.get('price_to_book'),
            ev_to_ebitda=fundamentals.get('ev_to_ebitda'),
            price_to_sales=fundamentals.get('price_to_sales'),
            valuation_score=float(valuation_score),
            valuation_signal=valuation_signal,
            revenue_growth=fundamentals.get('revenue_growth'),
            earnings_growth=fundamentals.get('earnings_growth'),
            free_cash_flow_growth=fundamentals.get('fcf_growth'),
            growth_score=float(growth_score),
            growth_signal=growth_signal,
            profit_margin=fundamentals.get('profit_margin'),
            operating_margin=fundamentals.get('operating_margin'),
            roe=fundamentals.get('roe'),
            roa=fundamentals.get('roa'),
            roic=fundamentals.get('roic'),
            gross_margin=fundamentals.get('gross_margin'),
            profitability_score=float(profitability_score),
            profitability_signal=profitability_signal,
            debt_to_equity=fundamentals.get('debt_to_equity'),
            current_ratio=fundamentals.get('current_ratio'),
            quick_ratio=fundamentals.get('quick_ratio'),
            interest_coverage=fundamentals.get('interest_coverage'),
            free_cash_flow=fundamentals.get('free_cash_flow'),
            health_score=float(health_score),
            health_signal=health_signal,
            dividend_yield=dividend_yield,
            payout_ratio=payout_ratio,
            buyback_yield=buyback_yield,
            shareholder_yield=shareholder_yield if shareholder_yield > 0 else None,
            asset_turnover=fundamentals.get('asset_turnover'),
            inventory_turnover=fundamentals.get('inventory_turnover'),
            receivables_turnover=fundamentals.get('receivables_turnover'),
            overall_score=float(overall_score),
            overall_signal=overall_signal,
            reasoning=reasoning,
            confidence=float(confidence),
            timestamp=datetime.now()
        )
    
    def _analyze_valuation(self, fundamentals: Dict) -> tuple:
        """
        Analyze valuation metrics
        
        Returns:
            (score, signal)
        """
        score = 0.5  # Default neutral
        positive_signals = 0
        total_checks = 0
        
        # P/E Ratio (lower is better)
        pe = fundamentals.get('pe_ratio')
        if pe is not None and pe > 0:
            total_checks += 1
            if pe < self.GOOD_PE:
                positive_signals += 1
                score += 0.1
        
        # PEG Ratio (lower is better, <1 is great)
        peg = fundamentals.get('peg_ratio')
        if peg is not None and peg > 0:
            total_checks += 1
            if peg < self.GOOD_PEG:
                positive_signals += 1
                score += 0.1
            if peg < 1.0:
                score += 0.1  # Bonus for PEG < 1
        
        # Price to Book (lower is generally better)
        pb = fundamentals.get('price_to_book')
        if pb is not None and pb > 0:
            total_checks += 1
            if pb < 3.0:  # Reasonable P/B
                positive_signals += 1
                score += 0.1
        
        # Normalize score
        if total_checks > 0:
            score = min(positive_signals / total_checks, 1.0)
        
        # Determine signal
        if score > 0.7:
            signal = "UNDERVALUED"
        elif score < 0.3:
            signal = "OVERVALUED"
        else:
            signal = "FAIR"
        
        return score, signal
    
    def _analyze_growth(self, fundamentals: Dict) -> tuple:
        """
        Analyze growth metrics
        
        Returns:
            (score, signal)
        """
        score = 0.5
        positive_signals = 0
        total_checks = 0
        
        # Revenue Growth
        rev_growth = fundamentals.get('revenue_growth')
        if rev_growth is not None:
            total_checks += 1
            if rev_growth > self.GOOD_REVENUE_GROWTH:
                positive_signals += 1
                score += 0.2
            if rev_growth > 0.20:  # Exceptional growth
                score += 0.1
        
        # Earnings Growth
        earn_growth = fundamentals.get('earnings_growth')
        if earn_growth is not None:
            total_checks += 1
            if earn_growth > self.GOOD_REVENUE_GROWTH:
                positive_signals += 1
                score += 0.2
            if earn_growth > 0.20:
                score += 0.1
        
        # Normalize
        if total_checks > 0:
            score = min(positive_signals / total_checks, 1.0)
        
        # Determine signal
        if score > 0.7:
            signal = "HIGH_GROWTH"
        elif score < 0.3:
            signal = "LOW_GROWTH"
        else:
            signal = "MODERATE"
        
        return score, signal
    
    def _analyze_profitability(self, fundamentals: Dict) -> tuple:
        """
        Analyze profitability metrics
        
        Returns:
            (score, signal)
        """
        score = 0.5
        positive_signals = 0
        total_checks = 0
        
        # Profit Margin
        profit_margin = fundamentals.get('profit_margin')
        if profit_margin is not None:
            total_checks += 1
            if profit_margin > self.GOOD_PROFIT_MARGIN:
                positive_signals += 1
                score += 0.15
        
        # Operating Margin
        op_margin = fundamentals.get('operating_margin')
        if op_margin is not None:
            total_checks += 1
            if op_margin > self.GOOD_PROFIT_MARGIN:
                positive_signals += 1
                score += 0.15
        
        # ROE (Return on Equity)
        roe = fundamentals.get('roe')
        if roe is not None:
            total_checks += 1
            if roe > self.GOOD_ROE:
                positive_signals += 1
                score += 0.2
            if roe > 0.25:  # Exceptional ROE
                score += 0.1
        
        # Normalize
        if total_checks > 0:
            score = min(positive_signals / total_checks, 1.0)
        
        # Determine signal
        if score > 0.7:
            signal = "HIGHLY_PROFITABLE"
        elif score < 0.3:
            signal = "STRUGGLING"
        else:
            signal = "PROFITABLE"
        
        return score, signal
    
    def _analyze_financial_health(self, fundamentals: Dict) -> tuple:
        """
        Analyze financial health metrics
        
        Returns:
            (score, signal)
        """
        score = 0.5
        positive_signals = 0
        total_checks = 0
        
        # Debt to Equity (lower is better)
        debt_to_equity = fundamentals.get('debt_to_equity')
        if debt_to_equity is not None:
            total_checks += 1
            if debt_to_equity < self.GOOD_DEBT_TO_EQUITY:
                positive_signals += 1
                score += 0.2
        
        # Current Ratio (higher is better, >1.5 is good)
        current_ratio = fundamentals.get('current_ratio')
        if current_ratio is not None:
            total_checks += 1
            if current_ratio > self.GOOD_CURRENT_RATIO:
                positive_signals += 1
                score += 0.15
        
        # Quick Ratio (>1.0 is good)
        quick_ratio = fundamentals.get('quick_ratio')
        if quick_ratio is not None:
            total_checks += 1
            if quick_ratio > 1.0:
                positive_signals += 1
                score += 0.15
        
        # Normalize
        if total_checks > 0:
            score = min(positive_signals / total_checks, 1.0)
        
        # Determine signal
        if score > 0.7:
            signal = "STRONG"
        elif score < 0.3:
            signal = "WEAK"
        else:
            signal = "STABLE"
        
        return score, signal
    
    def _determine_overall_signal(self, overall_score: float) -> str:
        """Determine overall investment signal"""
        if overall_score >= 0.8:
            return "STRONG_BUY"
        elif overall_score >= 0.6:
            return "BUY"
        elif overall_score >= 0.4:
            return "HOLD"
        else:
            return "AVOID"
    
    def _calculate_confidence(self, fundamentals: Dict) -> float:
        """
        Calculate confidence based on data availability
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        key_metrics = [
            'pe_ratio', 'revenue_growth', 'earnings_growth',
            'profit_margin', 'roe', 'debt_to_equity', 'current_ratio'
        ]
        
        available = sum(1 for m in key_metrics if fundamentals.get(m) is not None)
        confidence = available / len(key_metrics)
        
        return confidence
    
    def _generate_reasoning(
        self,
        fundamentals: Dict,
        valuation_signal: str,
        growth_signal: str,
        profitability_signal: str,
        health_signal: str
    ) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        # Valuation
        pe = fundamentals.get('pe_ratio')
        if pe:
            if valuation_signal == "UNDERVALUED":
                reasoning.append(f"Attractive valuation with P/E of {pe:.1f}")
            elif valuation_signal == "OVERVALUED":
                reasoning.append(f"Elevated valuation with P/E of {pe:.1f}")
            else:
                reasoning.append(f"Fair valuation with P/E of {pe:.1f}")
        
        # Growth
        rev_growth = fundamentals.get('revenue_growth')
        if rev_growth:
            reasoning.append(f"Revenue growth of {rev_growth:.1%} YoY - {growth_signal}")
        
        # Profitability
        profit_margin = fundamentals.get('profit_margin')
        roe = fundamentals.get('roe')
        if profit_margin:
            reasoning.append(f"Profit margin of {profit_margin:.1%} - {profitability_signal}")
        if roe:
            reasoning.append(f"ROE of {roe:.1%} - efficient capital usage")
        
        # Financial Health
        debt_to_equity = fundamentals.get('debt_to_equity')
        if debt_to_equity is not None:
            if health_signal == "STRONG":
                reasoning.append(f"Strong balance sheet with D/E ratio of {debt_to_equity:.2f}")
            elif health_signal == "WEAK":
                reasoning.append(f"High leverage with D/E ratio of {debt_to_equity:.2f}")
            else:
                reasoning.append(f"Stable balance sheet with D/E ratio of {debt_to_equity:.2f}")
        
        current_ratio = fundamentals.get('current_ratio')
        if current_ratio:
            reasoning.append(f"Current ratio of {current_ratio:.2f} - good liquidity")
        
        return reasoning


# Example usage
if __name__ == "__main__":
    # Initialize agent
    fa_agent = FundamentalAnalysisAgent()
    
    # Test with Apple
    symbol = "AAPL"
    
    print(f"\n{'='*60}")
    print(f"Fundamental Analysis: {symbol}")
    print(f"{'='*60}\n")
    
    try:
        score = fa_agent.analyze(symbol)
        
        print(f"💰 Valuation: {score.valuation_signal} (Score: {score.valuation_score:.2f})")
        if score.pe_ratio:
            print(f"  • P/E Ratio: {score.pe_ratio:.1f}")
        if score.peg_ratio:
            print(f"  • PEG Ratio: {score.peg_ratio:.2f}")
        
        print(f"\n📈 Growth: {score.growth_signal} (Score: {score.growth_score:.2f})")
        if score.revenue_growth:
            print(f"  • Revenue Growth: {score.revenue_growth:.1%}")
        if score.earnings_growth:
            print(f"  • Earnings Growth: {score.earnings_growth:.1%}")
        
        print(f"\n💵 Profitability: {score.profitability_signal} (Score: {score.profitability_score:.2f})")
        if score.profit_margin:
            print(f"  • Profit Margin: {score.profit_margin:.1%}")
        if score.roe:
            print(f"  • ROE: {score.roe:.1%}")
        
        print(f"\n🏦 Financial Health: {score.health_signal} (Score: {score.health_score:.2f})")
        if score.debt_to_equity is not None:
            print(f"  • Debt/Equity: {score.debt_to_equity:.2f}")
        if score.current_ratio:
            print(f"  • Current Ratio: {score.current_ratio:.2f}")
        
        print(f"\n⭐ Overall Assessment: {score.overall_signal}")
        print(f"   Overall Score: {score.overall_score:.2f}/1.00")
        print(f"   Confidence: {score.confidence:.1%}")
        
        print(f"\n💡 Key Insights:")
        for reason in score.reasoning:
            print(f"  • {reason}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"Error: {e}")
