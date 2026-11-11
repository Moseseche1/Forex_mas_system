"""Deterministic EMA and RSI calculations."""
from math import isfinite

def ema(values: list[float], period: int) -> list[float]:
    if period<1: raise ValueError("period must be >= 1")
    if not values: return []
    out=[float(values[0])]; alpha=2/(period+1)
    for raw in values[1:]:
        value=float(raw)
        if not isfinite(value): raise ValueError("prices must be finite")
        out.append(alpha*value+(1-alpha)*out[-1])
    return out

def rsi(values: list[float], period: int=14) -> list[float|None]:
    if period<1: raise ValueError("period must be >= 1")
    if any(not isfinite(float(v)) for v in values): raise ValueError("prices must be finite")
    out=[None]*len(values)
    if len(values)<=period: return out
    changes=[float(values[i])-float(values[i-1]) for i in range(1,len(values))]
    gains=[max(x,0) for x in changes]; losses=[max(-x,0) for x in changes]
    ag=sum(gains[:period])/period; al=sum(losses[:period])/period
    for i in range(period,len(values)):
        if i>period:
            ag=((period-1)*ag+gains[i-1])/period
            al=((period-1)*al+losses[i-1])/period
        out[i]=100.0 if al==0 else 100-100/(1+ag/al)
    return out

def ema_crossover(closes: list[float], fast: int=12, slow: int=26) -> list[int]:
    if fast<1 or slow<1 or fast>=slow: raise ValueError("Require 1 <= fast < slow")
    if len(closes)<slow: return [0]*len(closes)
    f=ema(closes,fast); s=ema(closes,slow); out=[0]*len(closes); prev=f[0]-s[0]
    for i in range(1,len(closes)):
        cur=f[i]-s[i]
        if prev<=0<cur: out[i]=1
        elif prev>=0>cur: out[i]=-1
        prev=cur
    return out
