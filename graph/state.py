"""
LangGraph State Management
Defines the shared state that flows through all agents
"""

from typing import TypedDict, Optional, List, Dict
from datetime import datetime
from agents.sentiment_agent import SentimentScore
from agents.technical_agent import TechnicalSignals
from agents.fundamental_agent import FundamentalScore
from agents.risk_agent import RiskAssessment


class TradingState(TypedDict, total=False):
    """
    Shared state that flows through the LangGraph workflow
    Each agent reads from and writes to this state
    """
    # Input
    symbol: str
    mode: str  # "SIMULATION", "APPROVAL_REQUIRED", "AUTO_EXECUTE"
    
    # Agent Outputs
    sentiment: Optional[SentimentScore]
    technical: Optional[TechnicalSignals]
    fundamental: Optional[FundamentalScore]
    risk: Optional[RiskAssessment]
    
    # Decision
    recommendation: Optional[str]  # "BUY", "HOLD", "SELL"
    confidence: Optional[float]
    reasoning: List[str]
    
    # Execution
    trade_approved: bool
    execution_status: Optional[str]
    order_id: Optional[str]
    
    # Metadata
    timestamp: datetime
    errors: List[str]
