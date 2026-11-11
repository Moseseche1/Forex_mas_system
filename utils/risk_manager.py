"""Legacy risk-manager compatibility layer backed by fail-closed position sizing."""
from dataclasses import dataclass
from threading import RLock
from datetime import datetime,timezone
from typing import Optional
from core.risk import size_position

@dataclass
class RiskParameters:
    max_daily_drawdown:float=.03
    max_trade_risk:float=.01
    max_portfolio_risk:float=.03
    max_position_size:float=.03
    min_trade_confidence:float=.60
    min_strategy_confidence:float=.70
    emergency_drawdown_level:float=.03
    max_consecutive_losses:int=3
    cooling_off_period:int=1800
    max_leverage:int=10
    min_position_size:float=.01
    def __post_init__(self):
        if not 0<self.max_trade_risk<=.01: raise ValueError("max_trade_risk must not exceed 1%")
        if not 0<self.max_daily_drawdown<=.03: raise ValueError("max_daily_drawdown must not exceed 3%")
        if not 0<self.max_portfolio_risk<=.03: raise ValueError("max_portfolio_risk must not exceed 3%")

class PositionSizer:
    def __init__(self,risk_params:RiskParameters):
        self.risk_params=risk_params
    def calculate_position_size(self,account_balance:float,current_drawdown:float,
        market_volatility:float,strategy_confidence:float,symbol_volatility:float,
        portfolio_correlation:float=0.0,*,entry_price:float|None=None,
        stop_price:float|None=None,tick_size:float|None=None,
        tick_value_per_lot:float|None=None,volume_min:float|None=None,
        volume_max:float|None=None,volume_step:float|None=None,estimated_cost_per_lot:float=0.0)->dict:
        blocked={"approved":False,"position_size":0.0,"volume_lots":0.0,
          "risk_percentage":0.0,"risk_amount":0.0,"reason":"Missing broker tick/stop metadata; fail closed"}
        if current_drawdown>=self.risk_params.max_daily_drawdown:
            blocked["reason"]="Daily drawdown limit reached"; return blocked
        if strategy_confidence<self.risk_params.min_trade_confidence:
            blocked["reason"]="Strategy confidence below threshold"; return blocked
        required=(entry_price,stop_price,tick_size,tick_value_per_lot,volume_min,volume_max,volume_step)
        if any(v is None for v in required): return blocked
        try:
            sized=size_position(equity=account_balance,risk_fraction=self.risk_params.max_trade_risk,
              entry_price=entry_price,stop_price=stop_price,tick_size=tick_size,
              tick_value_per_lot=tick_value_per_lot,volume_min=volume_min,
              volume_max=volume_max,volume_step=volume_step,estimated_cost_per_lot=estimated_cost_per_lot)
        except (ValueError,TypeError) as exc:
            blocked["reason"]=str(exc); return blocked
        return {"approved":sized.approved,"position_size":sized.volume_lots,
          "volume_lots":sized.volume_lots,"risk_percentage":(sized.estimated_loss/account_balance*100) if account_balance>0 else 0,
          "risk_amount":sized.estimated_loss,"reason":sized.reason,
          "risk_budget":sized.risk_budget}

class RiskMonitor:
    def __init__(self,risk_params:RiskParameters):
        self.risk_params=risk_params; self.lock=RLock()
        self.performance_history=[]; self.alert_history=[]; self.emergency_mode=False
        self.emergency_since=None
    def should_enter_trade(self,account_balance,current_drawdown,portfolio_exposure,
        market_conditions,strategy_metrics,open_positions)->dict:
        reasons=[]
        if account_balance<=0: reasons.append("Invalid account equity")
        if current_drawdown>=self.risk_params.max_daily_drawdown: reasons.append("Daily drawdown limit reached")
        if portfolio_exposure>=self.risk_params.max_portfolio_risk: reasons.append("Portfolio exposure limit reached")
        if strategy_metrics.get("confidence",0)<self.risk_params.min_strategy_confidence: reasons.append("Strategy confidence below threshold")
        if market_conditions.get("volatility",0)>.01: reasons.append("Volatility exceeds configured ceiling")
        if market_conditions.get("liquidity",1)<.5: reasons.append("Liquidity below configured threshold")
        if open_positions>=3: reasons.append("Maximum open position count reached")
        if self.emergency_mode: reasons.append("Emergency stop is active")
        return {"allowed":not reasons,"reasons":reasons,"warnings":[],
          "adjustments":{},"live_orders_sent":False}
    def update_performance(self,trade_result):
        with self.lock:
            self.performance_history.append(dict(trade_result))
            recent=self.performance_history[-self.risk_params.max_consecutive_losses:]
            if len(recent)>=self.risk_params.max_consecutive_losses and all(float(x.get("profit",0))<0 for x in recent):
                self.emergency_mode=True; self.emergency_since=datetime.now(timezone.utc)
                self.alert_history.append({"type":"consecutive_losses","timestamp":self.emergency_since.isoformat()})
    def release_emergency_mode(self):
        with self.lock: self.emergency_mode=False; self.emergency_since=None
    def get_risk_status(self):
        return {"emergency_mode":self.emergency_mode,"performance_history_size":len(self.performance_history),
          "recent_alerts":self.alert_history[-5:],"timestamp":datetime.now(timezone.utc).isoformat()}

class AdvancedRiskManager:
    def __init__(self,risk_params:Optional[RiskParameters]=None):
        self.risk_params=risk_params or RiskParameters()
        self.position_sizer=PositionSizer(self.risk_params)
        self.risk_monitor=RiskMonitor(self.risk_params)
        self.trade_history=[]
    def calculate_position_size(self,account_info,market_data,strategy_metrics):
        return self.position_sizer.calculate_position_size(
          account_balance=account_info.get("equity",account_info.get("balance",0)),
          current_drawdown=account_info.get("drawdown",1),
          market_volatility=market_data.get("volatility",0),
          strategy_confidence=strategy_metrics.get("confidence",0),
          symbol_volatility=market_data.get("symbol_volatility",0),
          entry_price=market_data.get("entry_price"),stop_price=market_data.get("stop_price"),
          tick_size=market_data.get("tick_size"),tick_value_per_lot=market_data.get("tick_value_per_lot"),
          volume_min=market_data.get("volume_min"),volume_max=market_data.get("volume_max"),
          volume_step=market_data.get("volume_step"),estimated_cost_per_lot=market_data.get("estimated_cost_per_lot",0.0))
    def should_enter_trade(self,account_info,market_data,strategy_metrics,open_positions):
        return self.risk_monitor.should_enter_trade(
          account_info.get("equity",account_info.get("balance",0)),
          account_info.get("drawdown",1),account_info.get("exposure",1),
          market_data,strategy_metrics,open_positions)
    def calculate_stop_loss_take_profit(self,entry_price,symbol_volatility,position_size,
        account_balance,timeframe="H1",direction="long"):
        if entry_price<=0 or symbol_volatility<=0 or direction not in {"long","short"}:
            raise ValueError("Valid entry, volatility and direction are required")
        multipliers={"M1":1.5,"M5":1.3,"M15":1.1,"M30":1.0,"H1":.9,"H4":.8,"D1":.7,"W1":.6}
        distance=symbol_volatility*2*multipliers.get(timeframe,1.0)
        sign=1 if direction=="long" else -1
        return {"stop_loss":entry_price-sign*distance,"take_profit":entry_price+sign*distance*2,
          "stop_distance":distance,"reward_risk_ratio":2.0}

# Compatibility singleton; fail-closed defaults are enforced by RiskParameters.
risk_manager=AdvancedRiskManager()
