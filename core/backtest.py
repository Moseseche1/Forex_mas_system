"""OHLC backtester. Signal at close i enters at next bar open; stop wins intrabar ties."""
from dataclasses import asdict,dataclass
from math import isfinite

@dataclass
class Trade:
    side:str; entry_index:int; exit_index:int; entry:float; exit:float
    volume_lots:float; gross_pnl:float; costs:float; net_pnl:float; exit_reason:str

def run_backtest(bars:list[dict],signals:list[int],*,initial_equity:float=1000.0,
 risk_fraction:float=.01,stop_distance:float=.002,reward_risk:float=2.0,
 tick_size:float=.0001,tick_value_per_lot:float=10.0,volume_min:float=.01,
 volume_max:float=100.0,volume_step:float=.01,spread:float=.0001,
 slippage:float=.00002,commission_per_lot:float=7.0)->dict:
    if len(bars)!=len(signals): raise ValueError("bars and signals must have equal lengths")
    if len(bars)<3: raise ValueError("At least three OHLC bars are required")
    if initial_equity<=0 or not 0<risk_fraction<=.01: raise ValueError("Invalid equity or risk fraction (max 1%)")
    if min(stop_distance,tick_size,tick_value_per_lot,volume_min,volume_step)<=0 or volume_max<volume_min: raise ValueError("Invalid stop or broker metadata")
    if reward_risk<=0 or min(spread,slippage,commission_per_lot)<0: raise ValueError("Invalid execution costs")
    for i,b in enumerate(bars):
        try: o,h,l,c=(float(b[k]) for k in ("open","high","low","close"))
        except (KeyError,TypeError,ValueError): raise ValueError(f"Bar {i} requires numeric OHLC")
        if not all(isfinite(x) and x>0 for x in (o,h,l,c)) or h<max(o,c) or l>min(o,c) or h<l: raise ValueError(f"Invalid OHLC at bar {i}")
        if signals[i] not in (-1,0,1): raise ValueError("Signals must be -1, 0, or 1")
    equity=initial_equity; peak=equity; maxdd=0.0; trades=[]; curve=[equity]; i=0
    while i<len(bars)-1:
        d=int(signals[i])
        if not d: i+=1; curve.append(equity); continue
        entry=(float(bars[i+1]["open"])+(spread/2+slippage)*d)
        stop=entry-d*stop_distance; target=entry+d*stop_distance*reward_risk
        budget=equity*risk_fraction; losslot=stop_distance/tick_size*tick_value_per_lot
        lots=min(volume_max,int((budget/losslot+1e-12)/volume_step)*volume_step)
        if lots<volume_min: i+=1; curve.append(equity); continue
        exitp=None; reason="end_of_data"; exitidx=len(bars)-1
        for j in range(i+1,len(bars)):
            hi=float(bars[j]["high"]); lo=float(bars[j]["low"])
            stop_hit=lo<=stop if d==1 else hi>=stop
            target_hit=hi>=target if d==1 else lo<=target
            if stop_hit: exitp=stop-d*(spread/2+slippage); reason="stop"; exitidx=j; break
            if target_hit: exitp=target-d*(spread/2+slippage); reason="target"; exitidx=j; break
        if exitp is None: exitp=float(bars[exitidx]["close"])-d*(spread/2+slippage)
        gross=d*(exitp-entry)/tick_size*tick_value_per_lot*lots
        costs=commission_per_lot*lots; net=gross-costs; equity+=net
        trades.append(Trade("long" if d==1 else "short",i+1,exitidx,entry,exitp,lots,gross,costs,net,reason))
        peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak if peak else 0)
        curve.append(equity); i=exitidx+1
    wins=[t for t in trades if t.net_pnl>0]; losses=[t for t in trades if t.net_pnl<0]
    gp=sum(t.net_pnl for t in wins); gl=abs(sum(t.net_pnl for t in losses))
    return {"initial_equity":initial_equity,"final_equity":round(equity,8),"net_profit":round(equity-initial_equity,8),
     "return_pct":round((equity/initial_equity-1)*100,6),"max_drawdown_pct":round(maxdd*100,6),
     "trade_count":len(trades),"win_rate_pct":round(len(wins)/len(trades)*100,4) if trades else 0.0,
     "profit_factor":round(gp/gl,6) if gl else (None if gp==0 else "infinite"),
     "trades":[asdict(t) for t in trades],"equity_curve":curve,
     "assumptions":{"signal_timing":"signal at close, entry next open","intrabar_conflict":"stop takes precedence",
     "spread":spread,"slippage":slippage,"commission_per_lot":commission_per_lot,"live_orders_sent":False}}
