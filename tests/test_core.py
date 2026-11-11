import pytest
from core.risk import size_position
from core.signals import ema,rsi,ema_crossover
from core.backtest import run_backtest
def test_position_risk_never_exceeds_one_percent():
 x=size_position(equity=100,risk_fraction=.01,entry_price=1.1,stop_price=1.099,tick_size=.0001,tick_value_per_lot=10,volume_min=.01,volume_max=100,volume_step=.01)
 assert x.approved and x.estimated_loss<=1+1e-9
def test_reject_minimum_lot_that_breaches_budget():
 x=size_position(equity=10,risk_fraction=.01,entry_price=1.1,stop_price=1.0,tick_size=.0001,tick_value_per_lot=10,volume_min=.01,volume_max=100,volume_step=.01)
 assert not x.approved and x.volume_lots==0
def test_reject_risk_above_one_percent():
 with pytest.raises(ValueError):
  size_position(equity=100,risk_fraction=.02,entry_price=1.1,stop_price=1.099,tick_size=.0001,tick_value_per_lot=10,volume_min=.01,volume_max=10,volume_step=.01)
def test_ema_and_rsi():
 assert len(ema([1,2,3],2))==3 and rsi(list(range(20)),14)[14]==100
def test_ema_crossover():
 out=ema_crossover([1,1,1,1,2,3,4,5],2,4)
 assert len(out)==8 and set(out)<={-1,0,1}
def test_backtest_next_bar_entry_and_costs():
 bars=[{"open":1.1,"high":1.101,"low":1.099,"close":1.1},{"open":1.1,"high":1.1045,"low":1.0995,"close":1.104},{"open":1.104,"high":1.105,"low":1.103,"close":1.104}]
 out=run_backtest(bars,[1,0,0],initial_equity=1000,stop_distance=.001,tick_size=.0001,tick_value_per_lot=10,volume_min=.01,volume_step=.01,spread=.0001,slippage=0,commission_per_lot=0)
 assert out["trade_count"]==1 and out["trades"][0]["entry_index"]==1
 assert out["assumptions"]["live_orders_sent"] is False
def test_backtest_rejects_invalid_ohlc():
 with pytest.raises(ValueError): run_backtest([{"open":1,"high":.9,"low":.8,"close":1}]*3,[0,0,0])

def test_risk_rejects_non_finite_values():
 with pytest.raises(ValueError):
  size_position(equity=float("nan"),risk_fraction=.01,entry_price=1.1,stop_price=1.099,tick_size=.0001,tick_value_per_lot=10,volume_min=.01,volume_max=100,volume_step=.01)

def test_legacy_risk_manager_fails_closed_without_broker_metadata():
 from utils.risk_manager import RiskParameters, PositionSizer
 s=PositionSizer(RiskParameters())
 result=s.calculate_position_size(account_balance=100,current_drawdown=0,
   market_volatility=.001,strategy_confidence=.9,symbol_volatility=.001)
 assert result["approved"] is False and result["position_size"]==0
