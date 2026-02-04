"""
Technical Analysis Agent
Analyzes stock price patterns, indicators, and trends using TA-Lib and pandas-ta
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel
import ta  # Technical Analysis library
from data.market_data import MarketDataProvider


class TechnicalSignals(BaseModel):
    """Structured technical analysis output"""
    symbol: str
    
    # RSI (Relative Strength Index)
    rsi: float
    rsi_signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    
    # MACD (Moving Average Convergence Divergence)
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    
    # Moving Averages
    sma_50: float
    sma_200: float
    price: float
    ma_signal: str  # "GOLDEN_CROSS", "DEATH_CROSS", "ABOVE_MA", "BELOW_MA"
    
    # Volume Analysis
    volume: int
    volume_sma_20: float
    volume_trend: str  # "INCREASING", "DECREASING", "NORMAL"
    
    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    
    # NEW: Stochastic Oscillator
    stoch_k: float
    stoch_d: float
    stoch_signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    
    # NEW: ADX (Average Directional Index) - Trend Strength
    adx: float
    adx_signal: str  # "STRONG_TREND", "WEAK_TREND", "NO_TREND"
    plus_di: float
    minus_di: float
    
    # NEW: ATR (Average True Range) - Volatility
    atr: float
    atr_percent: float  # ATR as % of price
    
    # NEW: Support & Resistance Levels
    support_level: float
    resistance_level: float
    price_to_support: float  # % distance to support
    price_to_resistance: float  # % distance to resistance
    
    # NEW: EMA (Exponential Moving Averages)
    ema_12: float
    ema_26: float
    
    # Overall Trend
    overall_trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    trend_strength: float  # 0.0 to 1.0
    
    # Reasoning
    reasoning: List[str]
    confidence: float
    timestamp: datetime


class TechnicalAnalysisAgent:
    """
    Analyzes technical indicators and price patterns
    Uses multiple indicators for robust signal generation
    """
    
    def __init__(self):
        """Initialize technical analysis agent"""
        self.market_data = MarketDataProvider()
        
        # Indicator thresholds
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30
        self.STOCH_OVERBOUGHT = 80
        self.STOCH_OVERSOLD = 20
        self.ADX_STRONG_TREND = 25
        self.ADX_WEAK_TREND = 20
        self.VOLUME_THRESHOLD = 1.5  # 50% above average
    
    def analyze(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> TechnicalSignals:
        """
        Perform comprehensive technical analysis
        
        Args:
            symbol: Stock ticker
            period: Historical data period
            interval: Data interval
            
        Returns:
            TechnicalSignals with all indicators
        """
        # Fetch historical data
        df = self.market_data.get_historical_data(
            symbol=symbol,
            period=period,
            interval=interval
        )
        
        if df.empty or len(df) < 50:
            raise ValueError(f"Insufficient data for technical analysis: {symbol}")
        
        # Calculate all indicators
        rsi_value, rsi_signal = self._calculate_rsi(df)
        macd_vals, macd_trend = self._calculate_macd(df)
        ma_vals, ma_signal = self._calculate_moving_averages(df)
        volume_vals, volume_trend = self._analyze_volume(df)
        bb_vals, bb_signal = self._calculate_bollinger_bands(df)
        
        # NEW: Calculate additional indicators
        stoch_vals, stoch_signal = self._calculate_stochastic(df)
        adx_vals, adx_signal = self._calculate_adx(df)
        atr_value, atr_percent = self._calculate_atr(df)
        support, resistance = self._calculate_support_resistance(df)
        ema_vals = self._calculate_ema(df)
        
        # Determine overall trend (now includes new indicators)
        overall_trend, trend_strength = self._determine_overall_trend(
            rsi_signal, macd_trend, ma_signal, volume_trend, bb_signal,
            stoch_signal, adx_signal
        )
        
        # Generate reasoning (enhanced with new indicators)
        reasoning = self._generate_reasoning(
            rsi_value, rsi_signal,
            macd_trend, ma_signal,
            volume_trend, bb_signal,
            overall_trend,
            stoch_vals, stoch_signal,
            adx_vals, adx_signal,
            atr_percent,
            support, resistance
        )
        
        # Calculate confidence (based on signal alignment)
        confidence = self._calculate_confidence(
            rsi_signal, macd_trend, ma_signal, volume_trend,
            stoch_signal, adx_signal
        )
        
        # Get current price
        current_price = df['Close'].iloc[-1]
        current_volume = int(df['Volume'].iloc[-1])
        
        return TechnicalSignals(
            symbol=symbol,
            rsi=float(rsi_value),
            rsi_signal=rsi_signal,
            macd=float(macd_vals['macd']),
            macd_signal=float(macd_vals['signal']),
            macd_histogram=float(macd_vals['histogram']),
            macd_trend=macd_trend,
            sma_50=float(ma_vals['sma_50']),
            sma_200=float(ma_vals['sma_200']),
            price=float(current_price),
            ma_signal=ma_signal,
            volume=current_volume,
            volume_sma_20=float(volume_vals['sma_20']),
            volume_trend=volume_trend,
            bb_upper=float(bb_vals['upper']),
            bb_middle=float(bb_vals['middle']),
            bb_lower=float(bb_vals['lower']),
            bb_signal=bb_signal,
            # NEW indicators
            stoch_k=float(stoch_vals['k']),
            stoch_d=float(stoch_vals['d']),
            stoch_signal=stoch_signal,
            adx=float(adx_vals['adx']),
            adx_signal=adx_signal,
            plus_di=float(adx_vals['plus_di']),
            minus_di=float(adx_vals['minus_di']),
            atr=float(atr_value),
            atr_percent=float(atr_percent),
            support_level=float(support),
            resistance_level=float(resistance),
            price_to_support=float((current_price - support) / current_price * 100),
            price_to_resistance=float((resistance - current_price) / current_price * 100),
            ema_12=float(ema_vals['ema_12']),
            ema_26=float(ema_vals['ema_26']),
            overall_trend=overall_trend,
            trend_strength=float(trend_strength),
            reasoning=reasoning,
            confidence=float(confidence),
            timestamp=datetime.now()
        )
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """
        Calculate RSI (Relative Strength Index)
        
        Returns:
            (rsi_value, signal)
        """
        rsi = ta.momentum.RSIIndicator(df['Close'], window=period)
        rsi_value = rsi.rsi().iloc[-1]
        
        if rsi_value > self.RSI_OVERBOUGHT:
            signal = "OVERBOUGHT"
        elif rsi_value < self.RSI_OVERSOLD:
            signal = "OVERSOLD"
        else:
            signal = "NEUTRAL"
        
        return rsi_value, signal
    
    def _calculate_macd(self, df: pd.DataFrame) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Returns:
            (macd_values_dict, trend)
        """
        macd = ta.trend.MACD(df['Close'])
        
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        histogram = macd.macd_diff().iloc[-1]
        
        # Determine trend
        if histogram > 0 and macd_line > signal_line:
            trend = "BULLISH"
        elif histogram < 0 and macd_line < signal_line:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        values = {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
        
        return values, trend
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> tuple:
        """
        Calculate Simple Moving Averages (50-day and 200-day)
        
        Returns:
            (ma_values_dict, signal)
        """
        sma_50 = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator().iloc[-1]
        sma_200 = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # Determine signal
        if sma_50 > sma_200 and current_price > sma_50:
            signal = "GOLDEN_CROSS"  # Bullish
        elif sma_50 < sma_200 and current_price < sma_50:
            signal = "DEATH_CROSS"  # Bearish
        elif current_price > sma_50 and current_price > sma_200:
            signal = "ABOVE_MA"  # Bullish
        elif current_price < sma_50 and current_price < sma_200:
            signal = "BELOW_MA"  # Bearish
        else:
            signal = "NEUTRAL"
        
        values = {
            'sma_50': sma_50,
            'sma_200': sma_200
        }
        
        return values, signal
    
    def _analyze_volume(self, df: pd.DataFrame) -> tuple:
        """
        Analyze volume trends
        
        Returns:
            (volume_values_dict, trend)
        """
        volume_sma_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        
        # Check if volume is significantly above average
        volume_ratio = current_volume / volume_sma_20
        
        if volume_ratio > self.VOLUME_THRESHOLD:
            trend = "INCREASING"
        elif volume_ratio < (1 / self.VOLUME_THRESHOLD):
            trend = "DECREASING"
        else:
            trend = "NORMAL"
        
        values = {
            'sma_20': volume_sma_20,
            'current': current_volume,
            'ratio': volume_ratio
        }
        
        return values, trend
    
    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        std_dev: int = 2
    ) -> tuple:
        """
        Calculate Bollinger Bands
        
        Returns:
            (bb_values_dict, signal)
        """
        bb = ta.volatility.BollingerBands(df['Close'], window=window, window_dev=std_dev)
        
        upper = bb.bollinger_hband().iloc[-1]
        middle = bb.bollinger_mavg().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # Determine signal based on price position
        if current_price >= upper:
            signal = "OVERBOUGHT"
        elif current_price <= lower:
            signal = "OVERSOLD"
        else:
            signal = "NEUTRAL"
        
        values = {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
        
        return values, signal
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        """
        Calculate Stochastic Oscillator
        
        Returns:
            (stoch_values_dict, signal)
        """
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'], 
            window=k_period, smooth_window=d_period
        )
        
        stoch_k = stoch.stoch().iloc[-1]
        stoch_d = stoch.stoch_signal().iloc[-1]
        
        # Determine signal
        if stoch_k > self.STOCH_OVERBOUGHT and stoch_d > self.STOCH_OVERBOUGHT:
            signal = "OVERBOUGHT"
        elif stoch_k < self.STOCH_OVERSOLD and stoch_d < self.STOCH_OVERSOLD:
            signal = "OVERSOLD"
        else:
            signal = "NEUTRAL"
        
        values = {'k': stoch_k, 'd': stoch_d}
        return values, signal
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """
        Calculate ADX (Average Directional Index) for trend strength
        
        Returns:
            (adx_values_dict, signal)
        """
        adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period)
        
        adx = adx_indicator.adx().iloc[-1]
        plus_di = adx_indicator.adx_pos().iloc[-1]
        minus_di = adx_indicator.adx_neg().iloc[-1]
        
        # Determine signal based on ADX value
        if adx >= self.ADX_STRONG_TREND:
            signal = "STRONG_TREND"
        elif adx >= self.ADX_WEAK_TREND:
            signal = "WEAK_TREND"
        else:
            signal = "NO_TREND"
        
        values = {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
        return values, signal
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """
        Calculate ATR (Average True Range) for volatility
        
        Returns:
            (atr_value, atr_as_percent_of_price)
        """
        atr_indicator = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=period)
        atr = atr_indicator.average_true_range().iloc[-1]
        
        current_price = df['Close'].iloc[-1]
        atr_percent = (atr / current_price) * 100
        
        return atr, atr_percent
    
    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> tuple:
        """
        Calculate Support and Resistance levels using recent highs/lows
        
        Returns:
            (support_level, resistance_level)
        """
        recent_data = df.tail(lookback)
        
        # Simple approach: use recent swing lows for support, swing highs for resistance
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        return support, resistance
    
    def _calculate_ema(self, df: pd.DataFrame) -> dict:
        """
        Calculate Exponential Moving Averages (12 and 26 period)
        
        Returns:
            dict with ema_12 and ema_26
        """
        ema_12 = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator().iloc[-1]
        ema_26 = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator().iloc[-1]
        
        return {'ema_12': ema_12, 'ema_26': ema_26}
    
    def _determine_overall_trend(
        self,
        rsi_signal: str,
        macd_trend: str,
        ma_signal: str,
        volume_trend: str,
        bb_signal: str,
        stoch_signal: str = "NEUTRAL",
        adx_signal: str = "NO_TREND"
    ) -> tuple:
        """
        Determine overall trend based on all indicators
        
        Returns:
            (overall_trend, strength)
        """
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 5
        
        # RSI
        if rsi_signal == "OVERSOLD":
            bullish_signals += 1
        elif rsi_signal == "OVERBOUGHT":
            bearish_signals += 1
        
        # MACD
        if macd_trend == "BULLISH":
            bullish_signals += 1
        elif macd_trend == "BEARISH":
            bearish_signals += 1
        
        # Moving Averages
        if ma_signal in ["GOLDEN_CROSS", "ABOVE_MA"]:
            bullish_signals += 1
        elif ma_signal in ["DEATH_CROSS", "BELOW_MA"]:
            bearish_signals += 1
        
        # Volume (increasing volume strengthens the trend)
        if volume_trend == "INCREASING":
            # Volume doesn't indicate direction, just strength
            pass
        
        # Bollinger Bands
        if bb_signal == "OVERSOLD":
            bullish_signals += 1
        elif bb_signal == "OVERBOUGHT":
            bearish_signals += 1
        
        # NEW: Stochastic Oscillator
        total_signals += 1
        if stoch_signal == "OVERSOLD":
            bullish_signals += 1
        elif stoch_signal == "OVERBOUGHT":
            bearish_signals += 1
        
        # ADX affects trend strength, not direction
        strength_multiplier = 1.0
        if adx_signal == "STRONG_TREND":
            strength_multiplier = 1.2
        elif adx_signal == "NO_TREND":
            strength_multiplier = 0.8
        
        # Determine overall trend
        if bullish_signals > bearish_signals:
            trend = "BULLISH"
            strength = (bullish_signals / total_signals) * strength_multiplier
        elif bearish_signals > bullish_signals:
            trend = "BEARISH"
            strength = (bearish_signals / total_signals) * strength_multiplier
        else:
            trend = "NEUTRAL"
            strength = 0.5
        
        # Cap strength at 1.0
        strength = min(strength, 1.0)
        
        return trend, strength
    
    def _calculate_confidence(
        self,
        rsi_signal: str,
        macd_trend: str,
        ma_signal: str,
        volume_trend: str,
        stoch_signal: str = "NEUTRAL",
        adx_signal: str = "NO_TREND"
    ) -> float:
        """
        Calculate confidence based on signal alignment
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        signals = [rsi_signal, macd_trend, ma_signal, stoch_signal]
        
        # Count how many signals agree
        bullish = sum(1 for s in signals if s in ["OVERSOLD", "BULLISH", "GOLDEN_CROSS", "ABOVE_MA"])
        bearish = sum(1 for s in signals if s in ["OVERBOUGHT", "BEARISH", "DEATH_CROSS", "BELOW_MA"])
        
        # High confidence if most signals agree
        max_agreement = max(bullish, bearish)
        total_signals = len(signals)
        
        base_confidence = max_agreement / total_signals
        
        # Boost confidence if volume is increasing (confirms trend)
        if volume_trend == "INCREASING":
            base_confidence *= 1.1
        
        # Boost confidence if ADX shows strong trend
        if adx_signal == "STRONG_TREND":
            base_confidence *= 1.1
        elif adx_signal == "NO_TREND":
            base_confidence *= 0.9
        
        # Cap at 1.0
        return min(base_confidence, 1.0)
    
    def _generate_reasoning(
        self,
        rsi_value: float,
        rsi_signal: str,
        macd_trend: str,
        ma_signal: str,
        volume_trend: str,
        bb_signal: str,
        overall_trend: str,
        stoch_vals: dict = None,
        stoch_signal: str = "NEUTRAL",
        adx_vals: dict = None,
        adx_signal: str = "NO_TREND",
        atr_percent: float = 0,
        support: float = 0,
        resistance: float = 0
    ) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        # RSI analysis
        if rsi_signal == "OVERBOUGHT":
            reasoning.append(f"RSI at {rsi_value:.1f} indicates overbought conditions - potential reversal")
        elif rsi_signal == "OVERSOLD":
            reasoning.append(f"RSI at {rsi_value:.1f} indicates oversold conditions - potential buying opportunity")
        else:
            reasoning.append(f"RSI at {rsi_value:.1f} shows neutral momentum")
        
        # MACD analysis
        if macd_trend == "BULLISH":
            reasoning.append("MACD shows bullish momentum (above signal line)")
        elif macd_trend == "BEARISH":
            reasoning.append("MACD shows bearish momentum (below signal line)")
        
        # Moving average analysis
        if ma_signal == "GOLDEN_CROSS":
            reasoning.append("Golden cross detected - strong bullish signal")
        elif ma_signal == "DEATH_CROSS":
            reasoning.append("Death cross detected - strong bearish signal")
        elif ma_signal == "ABOVE_MA":
            reasoning.append("Price trading above key moving averages - bullish")
        elif ma_signal == "BELOW_MA":
            reasoning.append("Price trading below key moving averages - bearish")
        
        # Volume analysis
        if volume_trend == "INCREASING":
            reasoning.append("Volume significantly above average - confirms trend strength")
        elif volume_trend == "DECREASING":
            reasoning.append("Volume below average - trend may be weakening")
        
        # Bollinger Bands
        if bb_signal == "OVERBOUGHT":
            reasoning.append("Price near upper Bollinger Band - potentially overbought")
        elif bb_signal == "OVERSOLD":
            reasoning.append("Price near lower Bollinger Band - potentially oversold")
        
        # NEW: Stochastic Oscillator
        if stoch_vals:
            if stoch_signal == "OVERBOUGHT":
                reasoning.append(f"Stochastic (%K: {stoch_vals['k']:.1f}, %D: {stoch_vals['d']:.1f}) shows overbought - momentum weakening")
            elif stoch_signal == "OVERSOLD":
                reasoning.append(f"Stochastic (%K: {stoch_vals['k']:.1f}, %D: {stoch_vals['d']:.1f}) shows oversold - potential bounce")
        
        # NEW: ADX Trend Strength
        if adx_vals:
            adx = adx_vals['adx']
            plus_di = adx_vals['plus_di']
            minus_di = adx_vals['minus_di']
            if adx_signal == "STRONG_TREND":
                direction = "bullish" if plus_di > minus_di else "bearish"
                reasoning.append(f"ADX at {adx:.1f} confirms strong {direction} trend (+DI: {plus_di:.1f}, -DI: {minus_di:.1f})")
            elif adx_signal == "NO_TREND":
                reasoning.append(f"ADX at {adx:.1f} indicates no clear trend - market consolidating")
        
        # NEW: ATR Volatility
        if atr_percent > 0:
            if atr_percent > 3:
                reasoning.append(f"ATR at {atr_percent:.2f}% of price - HIGH volatility, use wider stops")
            elif atr_percent < 1.5:
                reasoning.append(f"ATR at {atr_percent:.2f}% of price - LOW volatility, tighter range expected")
        
        # NEW: Support/Resistance
        if support > 0 and resistance > 0:
            reasoning.append(f"Support at ${support:.2f}, Resistance at ${resistance:.2f}")
        
        # Overall assessment
        reasoning.append(f"Overall technical outlook: {overall_trend}")
        
        return reasoning


# Example usage
if __name__ == "__main__":
    # Initialize agent
    ta_agent = TechnicalAnalysisAgent()
    
    # Test with Apple
    symbol = "AAPL"
    
    print(f"\n{'='*60}")
    print(f"Technical Analysis: {symbol}")
    print(f"{'='*60}\n")
    
    try:
        signals = ta_agent.analyze(symbol)
        
        print(f"📊 Price: ${signals.price:.2f}")
        print(f"\n🔍 Key Indicators:")
        print(f"  • RSI: {signals.rsi:.1f} ({signals.rsi_signal})")
        print(f"  • MACD: {signals.macd_trend}")
        print(f"  • Moving Averages: {signals.ma_signal}")
        print(f"  • Volume: {signals.volume_trend}")
        print(f"  • Bollinger Bands: {signals.bb_signal}")
        
        print(f"\n📈 Overall Assessment:")
        print(f"  • Trend: {signals.overall_trend}")
        print(f"  • Strength: {signals.trend_strength:.1%}")
        print(f"  • Confidence: {signals.confidence:.1%}")
        
        print(f"\n💡 Reasoning:")
        for reason in signals.reasoning:
            print(f"  • {reason}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"Error: {e}")
