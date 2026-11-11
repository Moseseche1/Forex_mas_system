"""Conservative lot sizing from broker tick economics."""
from dataclasses import dataclass
from math import floor, isfinite

@dataclass(frozen=True)
class SizingResult:
    approved: bool
    volume_lots: float
    risk_budget: float
    estimated_loss: float
    stop_distance: float
    reason: str

def size_position(*, equity: float, risk_fraction: float, entry_price: float,
                  stop_price: float, tick_size: float, tick_value_per_lot: float,
                  volume_min: float, volume_max: float, volume_step: float) -> SizingResult:
    vals=(equity,risk_fraction,entry_price,stop_price,tick_size,tick_value_per_lot,volume_min,volume_max,volume_step)
    if not all(isfinite(float(v)) for v in vals):
        raise ValueError("All sizing inputs must be finite")
    if equity<=0 or not 0<risk_fraction<=.01:
        raise ValueError("Equity must be positive and risk_fraction must be in (0, 1%]")
    if entry_price<=0 or stop_price<=0 or entry_price==stop_price:
        raise ValueError("Entry and stop must be positive and different")
    if min(tick_size,tick_value_per_lot,volume_min,volume_step)<=0 or volume_max<volume_min:
        raise ValueError("Invalid broker tick or volume metadata")
    distance=abs(entry_price-stop_price)
    budget=equity*risk_fraction
    loss_per_lot=(distance/tick_size)*tick_value_per_lot
    raw=budget/loss_per_lot
    volume=min(volume_max,floor((raw+1e-12)/volume_step)*volume_step)
    if volume+1e-12<volume_min:
        return SizingResult(False,0.0,budget,0.0,distance,"Minimum broker volume would exceed risk budget")
    loss=volume*loss_per_lot
    if loss>budget+max(1e-8,budget*1e-9):
        return SizingResult(False,0.0,budget,loss,distance,"Rounded volume breaches risk budget")
    return SizingResult(True,round(volume,8),budget,loss,distance,"Approved within configured risk limit")
