"""Explainable specialist-agent ensemble; produces research signals only."""
from math import sqrt
from core.signals import ema, rsi

def analyze_market(closes:list[float], *, risk_fraction:float=.01,
                   current_drawdown:float=0.0, max_bar_volatility:float=.01)->dict:
    if len(closes)<30: raise ValueError("At least 30 closing prices are required")
    if any(float(x)<=0 for x in closes): raise ValueError("Prices must be positive")
    if not 0<risk_fraction<=.01: raise ValueError("Risk per trade must be in (0, 1%]")
    if not 0<=current_drawdown<1: raise ValueError("Drawdown must be between 0 and 1")
    if max_bar_volatility<=0: raise ValueError("Volatility threshold must be positive")
    prices=[float(x) for x in closes]
    fast=ema(prices,12)[-1]; slow=ema(prices,26)[-1]
    trend="long" if fast>slow else "short" if fast<slow else "neutral"
    momentum_value=rsi(prices,14)[-1]
    momentum="long" if momentum_value is not None and momentum_value>=55 else "short" if momentum_value is not None and momentum_value<=45 else "neutral"
    returns=[prices[i]/prices[i-1]-1 for i in range(1,len(prices))]
    sample=returns[-20:]
    mean=sum(sample)/len(sample)
    volatility=sqrt(sum((x-mean)**2 for x in sample)/max(1,len(sample)-1))
    volatility_agent={"state":"normal" if volatility<=max_bar_volatility else "elevated",
                      "realized_bar_volatility":volatility,"trade_veto":volatility>max_bar_volatility}
    risk_agent={"state":"approved" if current_drawdown<.03 and risk_fraction<=.01 else "blocked",
                "risk_fraction":risk_fraction,"current_drawdown":current_drawdown,
                "trade_veto":current_drawdown>=.03 or risk_fraction>.01}
    votes=[trend,momentum]
    aligned=(votes.count("long")>=2 or votes.count("short")>=2)
    direction="long" if votes.count("long")>=2 else "short" if votes.count("short")>=2 else "neutral"
    veto=volatility_agent["trade_veto"] or risk_agent["trade_veto"]
    decision="no_trade" if veto or not aligned else direction
    confidence=(max(votes.count("long"),votes.count("short"))/len(votes)) if aligned else 0.0
    return {"decision":decision,"confidence":confidence,"agents":{
      "trend":{"state":trend,"ema_fast":fast,"ema_slow":slow},
      "momentum":{"state":momentum,"rsi_14":momentum_value},
      "volatility":volatility_agent,"risk":risk_agent,
      "portfolio":{"state":"consensus" if aligned else "no_consensus","votes":votes,"veto":veto},
      "executive":{"recommendation":decision,"explanation":"At least two directional research agents must agree; risk and volatility agents can veto."}},
      "execution":{"mode":"research_only","live_orders_sent":False}}
