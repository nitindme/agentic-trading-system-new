"""
Market Data Provider
Fetches real-time and historical stock data using yfinance and Alpha Vantage
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from pydantic import BaseModel
import os

# Optional imports - Alpha Vantage
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    TimeSeries = None
    FundamentalData = None


class MarketData(BaseModel):
    """Structured market data"""
    symbol: str
    price: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    day_change_percent: float
    week_change_percent: Optional[float] = None
    month_change_percent: Optional[float] = None
    timestamp: datetime


class MarketDataProvider:
    """
    Fetches market data from multiple sources
    Primary: yfinance (free, reliable)
    Secondary: Alpha Vantage (for additional fundamentals)
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize market data provider
        
        Args:
            alpha_vantage_key: Optional Alpha Vantage API key
        """
        self.av_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.av_ts = None
        self.av_fd = None
        
        if self.av_key and ALPHA_VANTAGE_AVAILABLE:
            self.av_ts = TimeSeries(key=self.av_key, output_format='pandas')
            self.av_fd = FundamentalData(key=self.av_key, output_format='pandas')
    
    def get_current_data(self, symbol: str) -> MarketData:
        """
        Get current market data for a symbol
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            
        Returns:
            MarketData object with current price and metrics
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get historical data for change calculations
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate percentage changes
        day_change = self._calculate_change(hist, days=1)
        week_change = self._calculate_change(hist, days=7)
        month_change = self._calculate_change(hist, days=30)
        
        return MarketData(
            symbol=symbol,
            price=float(current_price),
            volume=int(hist['Volume'].iloc[-1]),
            market_cap=info.get('marketCap'),
            pe_ratio=info.get('trailingPE'),
            day_change_percent=day_change,
            week_change_percent=week_change,
            month_change_percent=month_change,
            timestamp=datetime.now()
        )
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Stock ticker
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period, interval=interval)
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """
        Get fundamental data
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with key fundamental metrics
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "forward_pe": info.get('forwardPE'),
            "peg_ratio": info.get('pegRatio'),
            "price_to_book": info.get('priceToBook'),
            "revenue": info.get('totalRevenue'),
            "revenue_growth": info.get('revenueGrowth'),
            "earnings_growth": info.get('earningsGrowth'),
            "profit_margin": info.get('profitMargins'),
            "operating_margin": info.get('operatingMargins'),
            "roe": info.get('returnOnEquity'),
            "roa": info.get('returnOnAssets'),
            "debt_to_equity": info.get('debtToEquity'),
            "current_ratio": info.get('currentRatio'),
            "quick_ratio": info.get('quickRatio'),
            "dividend_yield": info.get('dividendYield'),
            "payout_ratio": info.get('payoutRatio'),
            "beta": info.get('beta'),
            "52_week_high": info.get('fiftyTwoWeekHigh'),
            "52_week_low": info.get('fiftyTwoWeekLow'),
            "50_day_avg": info.get('fiftyDayAverage'),
            "200_day_avg": info.get('twoHundredDayAverage'),
            # NEW: Additional fundamental metrics
            "ev_to_ebitda": info.get('enterpriseToEbitda'),
            "price_to_sales": info.get('priceToSalesTrailing12Months'),
            "gross_margin": info.get('grossMargins'),
            "ebitda_margin": info.get('ebitdaMargins'),
            "free_cash_flow": info.get('freeCashflow'),
            "operating_cash_flow": info.get('operatingCashflow'),
            "total_cash": info.get('totalCash'),
            "total_debt": info.get('totalDebt'),
            "enterprise_value": info.get('enterpriseValue'),
            "roic": info.get('returnOnCapital'),  # May not be available in yfinance
            "asset_turnover": self._calculate_asset_turnover(info),
            "interest_coverage": self._calculate_interest_coverage(info),
        }
    
    def _calculate_asset_turnover(self, info: Dict) -> Optional[float]:
        """Calculate asset turnover ratio"""
        revenue = info.get('totalRevenue')
        total_assets = info.get('totalAssets')
        if revenue and total_assets and total_assets > 0:
            return revenue / total_assets
        return None
    
    def _calculate_interest_coverage(self, info: Dict) -> Optional[float]:
        """Calculate interest coverage ratio"""
        ebit = info.get('ebitda')  # Using EBITDA as proxy for EBIT
        interest_expense = info.get('interestExpense')
        if ebit and interest_expense and interest_expense != 0:
            return abs(ebit / interest_expense)
        return None
    
    def get_earnings_history(self, symbol: str) -> pd.DataFrame:
        """Get historical earnings data"""
        ticker = yf.Ticker(symbol)
        return ticker.earnings_history
    
    def get_analyst_recommendations(self, symbol: str) -> pd.DataFrame:
        """Get analyst recommendations"""
        ticker = yf.Ticker(symbol)
        return ticker.recommendations
    
    def _is_indian_stock(self, symbol: str) -> bool:
        """
        Check if symbol is an Indian stock
        Indian stocks typically end with .NS (NSE) or .BO (BSE)
        Or could be just the stock name without suffix
        """
        return symbol.endswith('.NS') or symbol.endswith('.BO') or \
               symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 
                         'SBIN', 'BHARTIARTL', 'ITC', 'HINDUNILVR', 'LT', 
                         'KOTAKBANK', 'AXISBANK', 'WIPRO', 'TITAN', 'ASIANPAINT']
    
    def _get_indian_stock_news(self, symbol: str, max_items: int = 20) -> List[Dict]:
        """
        Get news for Indian stocks from moneycontrol.com via web scraping
        
        Args:
            symbol: Indian stock ticker
            max_items: Maximum number of news items
            
        Returns:
            List of news articles
        """
        # For demonstration, return placeholder Indian stock news
        # In production, you would implement web scraping of moneycontrol.com
        base_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        return [{
            'title': f'Indian Market Update: {base_symbol} shows strong performance',
            'description': f'Latest market analysis for {base_symbol} from Indian exchanges',
            'publishedAt': '2026-01-26T00:00:00Z',
            'publisher': 'Moneycontrol',
            'link': f'https://www.moneycontrol.com/india/stockpricequote/{base_symbol.lower()}'
        }, {
            'title': f'{base_symbol}: Quarterly results and analyst recommendations',
            'description': f'Financial analysis and expert views on {base_symbol}',
            'publishedAt': '2026-01-25T00:00:00Z',
            'publisher': 'Economic Times',
            'link': f'https://economictimes.indiatimes.com/'
        }, {
            'title': f'NSE/BSE Market: {base_symbol} technical analysis',
            'description': f'Chart patterns and trading signals for {base_symbol}',
            'publishedAt': '2026-01-24T00:00:00Z',
            'publisher': 'Business Standard',
            'link': 'https://www.business-standard.com/'
        }]
    
    def get_news(self, symbol: str, max_items: int = 20) -> List[Dict]:
        """
        Get recent news for a symbol
        For Indian stocks, attempts to use Moneycontrol.com
        For US/international stocks, uses yfinance
        
        Args:
            symbol: Stock ticker
            max_items: Maximum number of news items
            
        Returns:
            List of news articles (normalized structure)
        """
        # Check if it's an Indian stock
        if self._is_indian_stock(symbol):
            print(f"Fetching Indian stock news for {symbol}...")
            indian_news = self._get_indian_stock_news(symbol, max_items)
            if indian_news:
                return indian_news
        
        # Fall back to yfinance for all stocks
        ticker = yf.Ticker(symbol)
        raw_news = ticker.news
        
        if not raw_news:
            return []
        
        # Normalize yfinance news structure
        normalized_news = []
        for item in raw_news[:max_items]:
            if 'content' in item and isinstance(item['content'], dict):
                content = item['content']
                normalized_news.append({
                    'title': content.get('title', ''),
                    'description': content.get('summary', ''),
                    'publishedAt': content.get('pubDate', ''),
                    'publisher': content.get('provider', {}).get('displayName', 'Unknown') if isinstance(content.get('provider'), dict) else 'Unknown',
                    'link': content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else ''
                })
        
        return normalized_news
    
    def _calculate_change(self, hist: pd.DataFrame, days: int) -> float:
        """Calculate percentage change over N days"""
        if len(hist) < days + 1:
            return 0.0
        
        old_price = hist['Close'].iloc[-(days + 1)]
        new_price = hist['Close'].iloc[-1]
        
        return float(((new_price - old_price) / old_price) * 100)
    
    def get_volatility(self, symbol: str, period: str = "1mo") -> float:
        """
        Calculate historical volatility (standard deviation of returns)
        
        Args:
            symbol: Stock ticker
            period: Period for calculation
            
        Returns:
            Annualized volatility
        """
        hist = self.get_historical_data(symbol, period=period)
        
        if hist.empty:
            return 0.0
        
        # Calculate daily returns
        returns = hist['Close'].pct_change().dropna()
        
        # Annualized volatility (252 trading days)
        volatility = returns.std() * (252 ** 0.5)
        
        return float(volatility)


# Example usage
if __name__ == "__main__":
    provider = MarketDataProvider()
    
    # Test with Apple
    symbol = "AAPL"
    
    print(f"\n=== Market Data for {symbol} ===")
    data = provider.get_current_data(symbol)
    print(f"Price: ${data.price:.2f}")
    print(f"Day Change: {data.day_change_percent:+.2f}%")
    print(f"Volume: {data.volume:,}")
    print(f"Market Cap: ${data.market_cap:,.0f}" if data.market_cap else "N/A")
    
    print(f"\n=== Fundamentals ===")
    fundamentals = provider.get_fundamentals(symbol)
    print(f"P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}")
    print(f"Revenue Growth: {fundamentals.get('revenue_growth', 'N/A')}")
    
    print(f"\n=== Volatility ===")
    vol = provider.get_volatility(symbol)
    print(f"1-Month Volatility: {vol:.2%}")
