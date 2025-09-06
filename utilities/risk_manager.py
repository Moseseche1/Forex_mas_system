import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import math
from collections import deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    """Risk management parameters and limits"""
    # Core risk limits
    max_daily_drawdown: float = 0.05  # 5% max daily drawdown
    max_trade_risk: float = 0.02      # 2% max per trade
    max_portfolio_risk: float = 0.10  # 10% max total exposure
    max_position_size: float = 0.10   # 10% of portfolio per position
    
    # Volatility adjustments
    volatility_multiplier: float = 1.5
    max_volatility_adjustment: float = 0.5
    
    # Confidence thresholds
    min_trade_confidence: float = 0.6
    min_strategy_confidence: float = 0.7
    
    # Emergency controls
    emergency_drawdown_level: float = 0.08  # 8% drawdown triggers emergency
    max_consecutive_losses: int = 5         # 5 consecutive losses trigger review
    cooling_off_period: int = 300           # 5 minutes cooling off after emergency
    
    # Broker-specific limits
    max_leverage: int = 10
    min_position_size: float = 0.01  # Minimum position size

class PositionSizer:
    """Advanced position sizing calculator"""
    
    def __init__(self, risk_params: RiskParameters):
        self.risk_params = risk_params
        self.volatility_cache = {}
        self.correlation_matrix = {}
        
    def calculate_position_size(
        self,
        account_balance: float,
        current_drawdown: float,
        market_volatility: float,
        strategy_confidence: float,
        symbol_volatility: float,
        portfolio_correlation: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate dynamic position size with multiple risk factors
        """
        try:
            # Base position size from risk per trade
            base_size = account_balance * self.risk_params.max_trade_risk
            
            # Drawdown adjustment - reduce size during drawdown
            drawdown_penalty = self._calculate_drawdown_penalty(current_drawdown)
            
            # Volatility adjustment - reduce size in volatile markets
            volatility_penalty = self._calculate_volatility_penalty(
                market_volatility, 
                symbol_volatility
            )
            
            # Confidence adjustment - scale with strategy confidence
            confidence_factor = self._calculate_confidence_factor(strategy_confidence)
            
            # Correlation adjustment - reduce size for correlated positions
            correlation_penalty = self._calculate_correlation_penalty(portfolio_correlation)
            
            # Combined risk adjustment
            risk_adjustment = (
                drawdown_penalty * 
                volatility_penalty * 
                confidence_factor * 
                correlation_penalty
            )
            
            # Apply minimum and maximum limits
            position_size = base_size * risk_adjustment
            position_size = self._apply_position_limits(position_size, account_balance)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                position_size, 
                account_balance,
                symbol_volatility
            )
            
            logger.info(
                f"Position sizing: Base=${base_size:.2f}, "
                f"DrawdownAdj={drawdown_penalty:.3f}, "
                f"VolAdj={volatility_penalty:.3f}, "
                f"ConfAdj={confidence_factor:.3f}, "
                f"CorrAdj={correlation_penalty:.3f}, "
                f"Final=${position_size:.2f}, "
                f"RiskReward={risk_metrics['risk_reward_ratio']:.2f}"
            )
            
            return {
                'position_size': position_size,
                'risk_percentage': (position_size / account_balance) * 100,
                'risk_metrics': risk_metrics,
                'adjustment_factors': {
                    'drawdown_penalty': drawdown_penalty,
                    'volatility_penalty': volatility_penalty,
                    'confidence_factor': confidence_factor,
                    'correlation_penalty': correlation_penalty
                }
            }
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            return self._get_fallback_position_size(account_balance)
    
    def _calculate_drawdown_penalty(self, current_drawdown: float) -> float:
        """Calculate drawdown-based position size reduction"""
        if current_drawdown >= self.risk_params.max_daily_drawdown:
            return 0.0  # No trading at max drawdown
        
        # Linear reduction from 1.0 to 0.1 as drawdown increases
        penalty = 1.0 - (current_drawdown / self.risk_params.max_daily_drawdown)
        return max(0.1, min(1.0, penalty))
    
    def _calculate_volatility_penalty(self, market_volatility: float, symbol_volatility: float) -> float:
        """Calculate volatility-based position size reduction"""
        # Combine market and symbol volatility
        combined_volatility = (market_volatility + symbol_volatility) / 2
        
        # Exponential decay based on volatility
        penalty = math.exp(-self.risk_params.volatility_multiplier * combined_volatility)
        return max(0.1, min(1.0, penalty))
    
    def _calculate_confidence_factor(self, strategy_confidence: float) -> float:
        """Calculate confidence-based position size adjustment"""
        if strategy_confidence < self.risk_params.min_trade_confidence:
            return 0.0  # No trading below minimum confidence
        
        # Scale from 0.1 to 1.0 based on confidence
        factor = (strategy_confidence - self.risk_params.min_trade_confidence) / (1.0 - self.risk_params.min_trade_confidence)
        return max(0.1, min(1.0, factor))
    
    def _calculate_correlation_penalty(self, portfolio_correlation: float) -> float:
        """Calculate correlation-based position size reduction"""
        # Reduce position size for highly correlated assets
        correlation_penalty = 1.0 - abs(portfolio_correlation)
        return max(0.5, min(1.0, correlation_penalty))
    
    def _apply_position_limits(self, position_size: float, account_balance: float) -> float:
        """Apply minimum and maximum position size limits"""
        min_size = account_balance * 0.001  # 0.1% minimum
        max_size = account_balance * self.risk_params.max_position_size
        
        position_size = max(min_size, min(max_size, position_size))
        return round(position_size, 2)
    
    def _calculate_risk_metrics(self, position_size: float, account_balance: float, 
                              symbol_volatility: float) -> Dict[str, float]:
        """Calculate risk metrics for the position"""
        risk_amount = position_size
        risk_percentage = (risk_amount / account_balance) * 100
        
        # Estimate potential profit/loss based on volatility
        estimated_pl = position_size * symbol_volatility
        
        # Risk-reward ratio (simplified)
        risk_reward_ratio = 2.0  # Default 1:2
        
        return {
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'estimated_pl': estimated_pl,
            'risk_reward_ratio': risk_reward_ratio,
            'volatility_impact': symbol_volatility * 100
        }
    
    def _get_fallback_position_size(self, account_balance: float) -> Dict[str, float]:
        """Get fallback position size when calculation fails"""
        fallback_size = account_balance * 0.01  # 1% as fallback
        
        return {
            'position_size': fallback_size,
            'risk_percentage': 1.0,
            'risk_metrics': {
                'risk_amount': fallback_size,
                'risk_percentage': 1.0,
                'estimated_pl': fallback_size * 0.1,
                'risk_reward_ratio': 1.5,
                'volatility_impact': 0.1
            },
            'adjustment_factors': {
                'drawdown_penalty': 1.0,
                'volatility_penalty': 1.0,
                'confidence_factor': 1.0,
                'correlation_penalty': 1.0
            },
            'fallback': True
        }

class RiskMonitor:
    """Real-time risk monitoring and alerting system"""
    
    def __init__(self, risk_params: RiskParameters):
        self.risk_params = risk_params
        self.performance_history = deque(maxlen=100)
        self.drawdown_history = deque(maxlen=24)  # 24 hours
        self.alert_history = deque(maxlen=50)
        self.lock = threading.Lock()
        self.emergency_mode = False
        self.emergency_since = None
        
    def should_enter_trade(
        self,
        account_balance: float,
        current_drawdown: float,
        portfolio_exposure: float,
        market_conditions: Dict,
        strategy_metrics: Dict,
        open_positions: int
    ) -> Dict[str, any]:
        """
        Comprehensive trade entry decision with risk checks
        """
        with self.lock:
            decision = {
                'allowed': True,
                'reasons': [],
                'warnings': [],
                'adjustments': {}
            }
            
            # Hard risk limits
            if not self._check_hard_limits(account_balance, current_drawdown, portfolio_exposure, decision):
                decision['allowed'] = False
                return decision
            
            # Strategy confidence checks
            if not self._check_strategy_confidence(strategy_metrics, decision):
                decision['allowed'] = False
                return decision
            
            # Market condition checks
            if not self._check_market_conditions(market_conditions, decision):
                decision['allowed'] = False
                return decision
            
            # Position limits
            if not self._check_position_limits(open_positions, portfolio_exposure, decision):
                decision['allowed'] = False
                return decision
            
            # Emergency mode check
            if self.emergency_mode:
                if not self._check_emergency_override(decision):
                    decision['allowed'] = False
                    return decision
            
            return decision
    
    def _check_hard_limits(self, account_balance: float, current_drawdown: float, 
                          portfolio_exposure: float, decision: Dict) -> bool:
        """Check hard risk limits"""
        # Drawdown limit
        if current_drawdown >= self.risk_params.max_daily_drawdown:
            decision['reasons'].append(f"Daily drawdown limit reached: {current_drawdown:.2%}")
            return False
        
        # Portfolio exposure limit
        if portfolio_exposure >= self.risk_params.max_portfolio_risk:
            decision['reasons'].append(f"Portfolio exposure limit reached: {portfolio_exposure:.2%}")
            return False
        
        # Emergency drawdown level
        if current_drawdown >= self.risk_params.emergency_drawdown_level:
            decision['warnings'].append(f"Approaching emergency drawdown level: {current_drawdown:.2%}")
            self._trigger_emergency_mode("High drawdown")
        
        return True
    
    def _check_strategy_confidence(self, strategy_metrics: Dict, decision: Dict) -> bool:
        """Check strategy confidence levels"""
        confidence = strategy_metrics.get('confidence', 0)
        
        if confidence < self.risk_params.min_trade_confidence:
            decision['reasons'].append(f"Strategy confidence too low: {confidence:.2f}")
            return False
        
        if confidence < self.risk_params.min_strategy_confidence:
            decision['warnings'].append(f"Low strategy confidence: {confidence:.2f}")
            decision['adjustments']['confidence_reduction'] = 0.5
        
        return True
    
    def _check_market_conditions(self, market_conditions: Dict, decision: Dict) -> bool:
        """Check market conditions"""
        volatility = market_conditions.get('volatility', 0)
        liquidity = market_conditions.get('liquidity', 1)
        
        # High volatility check
        if volatility > 0.5:
            decision['reasons'].append(f"Market volatility too high: {volatility:.2f}")
            return False
        elif volatility > 0.3:
            decision['warnings'].append(f"High market volatility: {volatility:.2f}")
            decision['adjustments']['volatility_reduction'] = 0.7
        
        # Low liquidity check
        if liquidity < 0.3:
            decision['reasons'].append(f"Market liquidity too low: {liquidity:.2f}")
            return False
        elif liquidity < 0.6:
            decision['warnings'].append(f"Low market liquidity: {liquidity:.2f}")
            decision['adjustments']['liquidity_reduction'] = 0.8
        
        return True
    
    def _check_position_limits(self, open_positions: int, portfolio_exposure: float, 
                              decision: Dict) -> bool:
        """Check position limits"""
        # Too many open positions
        if open_positions > 10:
            decision['reasons'].append(f"Too many open positions: {open_positions}")
            return False
        elif open_positions > 5:
            decision['warnings'].append(f"High number of open positions: {open_positions}")
            decision['adjustments']['position_reduction'] = 0.6
        
        # High exposure with many positions
        if open_positions > 3 and portfolio_exposure > 0.05:
            decision['warnings'].append("High exposure with multiple positions")
            decision['adjustments']['exposure_reduction'] = 0.5
        
        return True
    
    def _check_emergency_override(self, decision: Dict) -> bool:
        """Check if trading is allowed during emergency mode"""
        # Allow only very conservative trades during emergency
        decision['warnings'].append("Trading in emergency mode - reduced size only")
        decision['adjustments']['emergency_reduction'] = 0.3
        return True
    
    def _trigger_emergency_mode(self, reason: str):
        """Trigger emergency risk management mode"""
        with self.lock:
            if not self.emergency_mode:
                self.emergency_mode = True
                self.emergency_since = datetime.utcnow()
                logger.critical(f"EMERGENCY MODE ACTIVATED: {reason}")
                
                # Log alert
                self.alert_history.append({
                    'type': 'emergency_mode',
                    'reason': reason,
                    'timestamp': datetime.utcnow(),
                    'action': 'activated'
                })
    
    def release_emergency_mode(self):
        """Release emergency mode"""
        with self.lock:
            if self.emergency_mode:
                self.emergency_mode = False
                duration = (datetime.utcnow() - self.emergency_since).total_seconds()
                logger.info(f"Emergency mode released after {duration:.0f} seconds")
                
                # Log alert
                self.alert_history.append({
                    'type': 'emergency_mode',
                    'reason': 'manual_release',
                    'timestamp': datetime.utcnow(),
                    'action': 'released',
                    'duration_seconds': duration
                })
    
    def update_performance(self, trade_result: Dict):
        """Update performance history and check for patterns"""
        with self.lock:
            self.performance_history.append(trade_result)
            
            # Check for consecutive losses
            recent_trades = list(self.performance_history)[-self.risk_params.max_consecutive_losses:]
            if len(recent_trades) >= self.risk_params.max_consecutive_losses:
                losses = sum(1 for trade in recent_trades if trade.get('profit', 0) < 0)
                if losses >= self.risk_params.max_consecutive_losses:
                    self._trigger_emergency_mode(f"{losses} consecutive losses")
    
    def update_drawdown(self, drawdown: float):
        """Update drawdown history"""
        with self.lock:
            self.drawdown_history.append({
                'drawdown': drawdown,
                'timestamp': datetime.utcnow()
            })
    
    def get_risk_status(self) -> Dict[str, any]:
        """Get current risk status"""
        with self.lock:
            return {
                'emergency_mode': self.emergency_mode,
                'emergency_since': self.emergency_since,
                'performance_history_size': len(self.performance_history),
                'drawdown_history_size': len(self.drawdown_history),
                'recent_alerts': list(self.alert_history)[-5:],
                'timestamp': datetime.utcnow().isoformat()
            }

class AdvancedRiskManager:
    """Main risk management orchestrator"""
    
    def __init__(self, risk_params: Optional[RiskParameters] = None):
        self.risk_params = risk_params or RiskParameters()
        self.position_sizer = PositionSizer(self.risk_params)
        self.risk_monitor = RiskMonitor(self.risk_params)
        self.trade_history = []
        
    def calculate_position_size(self, account_info: Dict, market_data: Dict, 
                               strategy_metrics: Dict) -> Dict[str, any]:
        """Calculate position size with all risk factors"""
        account_balance = account_info.get('balance', 10000)
        current_drawdown = account_info.get('drawdown', 0)
        portfolio_exposure = account_info.get('exposure', 0)
        
        market_volatility = market_data.get('volatility', 0.1)
        symbol_volatility = market_data.get('symbol_volatility', 0.1)
        strategy_confidence = strategy_metrics.get('confidence', 0.5)
        
        return self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            current_drawdown=current_drawdown,
            market_volatility=market_volatility,
            strategy_confidence=strategy_confidence,
            symbol_volatility=symbol_volatility
        )
    
    def should_enter_trade(self, account_info: Dict, market_data: Dict, 
                          strategy_metrics: Dict, open_positions: int) -> Dict[str, any]:
        """Determine if trade should be entered"""
        account_balance = account_info.get('balance', 10000)
        current_drawdown = account_info.get('drawdown', 0)
        portfolio_exposure = account_info.get('exposure', 0)
        
        return self.risk_monitor.should_enter_trade(
            account_balance=account_balance,
            current_drawdown=current_drawdown,
            portfolio_exposure=portfolio_exposure,
            market_conditions=market_data,
            strategy_metrics=strategy_metrics,
            open_positions=open_positions
        )
    
    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        symbol_volatility: float,
        position_size: float,
        account_balance: float,
        timeframe: str = 'H1'
    ) -> Dict[str, float]:
        """Dynamic stop loss and take profit calculation"""
        # Base stop loss from volatility
        atr_multiplier = 2.0
        stop_loss_distance = symbol_volatility * atr_multiplier
        
        # Adjust for timeframe
        timeframe_multiplier = {
            'M1': 1.5, 'M5': 1.3, 'M15': 1.1, 'M30': 1.0,
            'H1': 0.9, 'H4': 0.8, 'D1': 0.7, 'W1': 0.6
        }.get(timeframe, 1.0)
        
        stop_loss_distance *= timeframe_multiplier
        
        # Calculate stop loss and take profit
        stop_loss = entry_price - stop_loss_distance
        take_profit = entry_price + (stop_loss_distance * 2)  # 1:2 risk-reward
        
        # Ensure minimum distance
        stop_loss = max(0.0001, stop_loss)
        take_profit = max(0.0001, take_profit)
        
        # Calculate risk amount
        risk_amount = abs(entry_price - stop_loss) * position_size
        risk_percentage = (risk_amount / account_balance) * 100
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'risk_reward_ratio': 2.0,
            'volatility_factor': symbol_volatility
        }
    
    def emergency_stop_check(self, account_info: Dict, trade_history: List) -> bool:
        """Check if emergency stop conditions are met"""
        current_balance = account_info.get('balance', 10000)
        initial_balance = account_info.get('initial_balance', 10000)
        current_drawdown = 1 - (current_balance / initial_balance)
        
        # Critical drawdown
        if current_drawdown >= 0.10:  # 10% total drawdown
            logger.critical(f"EMERGENCY STOP: 10% total drawdown reached: {current_drawdown:.2%}")
            return True
        
        # Too many simultaneous losses
        recent_trades = trade_history[-10:] if len(trade_history) > 10 else trade_history
        recent_losses = sum(1 for trade in recent_trades if trade.get('profit', 0) < 0)
        
        if recent_losses >= 8:  # 8 losses in last 10 trades
            logger.critical(f"EMERGENCY STOP: {recent_losses} losses in last 10 trades")
            return True
        
        return False
    
    def update_trade_result(self, trade_result: Dict):
        """Update risk system with trade results"""
        self.trade_history.append(trade_result)
        self.risk_monitor.update_performance(trade_result)
    
    def update_drawdown(self, drawdown: float):
        """Update current drawdown"""
        self.risk_monitor.update_drawdown(drawdown)
    
    def release_emergency_mode(self):
        """Release emergency risk mode"""
        self.risk_monitor.release_emergency_mode()
    
    def get_risk_report(self) -> Dict[str, any]:
        """Get comprehensive risk report"""
        return {
            'parameters': self.risk_params.__dict__,
            'position_sizer': self.position_sizer.__class__.__name__,
            'risk_monitor': self.risk_monitor.get_risk_status(),
            'trade_history_count': len(self.trade_history),
            'current_time': datetime.utcnow().isoformat()
        }

# Global instance
risk_manager = AdvancedRiskManager()
