import time
import random
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def generate_market_signal() -> Dict[str, float]:
    """
    Generate simulated market signals for trading decisions
    """
    return {
        'trend_strength': random.uniform(0.1, 0.9),
        'volatility': random.uniform(0.05, 0.25),
        'momentum': random.uniform(-0.2, 0.2),
        'volume_ratio': random.uniform(0.5, 1.5),
        'signal_confidence': random.uniform(0.3, 0.95)
    }

def calculate_position_size(balance: float, risk_per_trade: float, 
                          stop_loss_pips: float, pip_value: float = 10.0) -> float:
    """
    Calculate position size based on risk management
    """
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return round(position_size, 2)

def emotional_intelligence_update(current_emotions: Dict[str, float], 
                                profit_loss: float, 
                                market_volatility: float) -> Dict[str, float]:
    """
    Update emotional state based on performance and market conditions
    """
    new_emotions = current_emotions.copy()
    
    # Confidence increases with profits, decreases with losses
    confidence_change = profit_loss * 0.15
    new_emotions['confidence'] = max(0.1, min(0.9, 
        new_emotions['confidence'] + confidence_change))
    
    # Caution increases with volatility and losses
    caution_change = market_volatility * 0.2 - profit_loss * 0.1
    new_emotions['caution'] = max(0.1, min(0.9, 
        new_emotions['caution'] + caution_change))
    
    # Aggression is confidence minus caution
    new_emotions['aggression'] = max(0.1, min(0.9, 
        new_emotions['confidence'] - new_emotions['caution']))
    
    return new_emotions

def format_currency(amount: float) -> str:
    """
    Format currency amount for display
    """
    return f"${amount:,.2f}"

def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for performance measurement
    """
    if not returns:
        return 0.0
    
    excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
    if len(excess_returns) < 2:
        return 0.0
    
    sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
    return round(sharpe, 2)

def should_trade_based_on_emotions(emotions: Dict[str, float]) -> bool:
    """
    Determine if trading should proceed based on emotional state
    """
    confidence = emotions.get('confidence', 0.5)
    caution = emotions.get('caution', 0.5)
    
    # Don't trade if too cautious or not confident enough
    if caution > 0.7 or confidence < 0.3:
        return False
    
    # Trade if confident and not too cautious
    return confidence > 0.4 and caution < 0.6

def generate_trade_id() -> str:
    """
    Generate unique trade ID
    """
    timestamp = int(time.time() * 1000)
    random_suffix = random.randint(1000, 9999)
    return f"TRADE_{timestamp}_{random_suffix}"
