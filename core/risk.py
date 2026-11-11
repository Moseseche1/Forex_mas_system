"""Conservative lot sizing using decimal broker tick economics."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from math import isfinite

@dataclass(frozen=True)
class SizingResult:
    approved: bool
    volume_lots: float
    risk_budget: float
    estimated_loss: float
    stop_distance: float
    reason: str

def size_position(*, equity:float, risk_fraction:float, entry_price:float,
                  stop_price:float, tick_size:float, tick_value_per_lot:float,
                  volume_min:float, volume_max:float, volume_step:float)->SizingResult:
    values=(equity,risk_fraction,entry_price,stop_price,tick_size,tick_value_per_lot,volume_min,volume_max,volume_step)
    if not all(isfinite(float(v)) for v in values):
        raise ValueError("All sizing inputs must be finite")
    if equity<=0 or not 0<risk_fraction<=.01:
        raise ValueError("Equity must be positive and risk_fraction must be in (0, 1%]")
    if entry_price<=0 or stop_price<=0 or entry_price==stop_price:
        raise ValueError("Entry and stop must be positive and different")
    if min(tick_size,tick_value_per_lot,volume_min,volume_step)<=0 or volume_max<volume_min:
        raise ValueError("Invalid broker tick or volume metadata")

    D=Decimal
    distance=abs(D(str(entry_price))-D(str(stop_price)))
    budget=D(str(equity))*D(str(risk_fraction))
    tick=D(str(tick_size)); tick_value=D(str(tick_value_per_lot))
    step=D(str(volume_step)); minimum=D(str(volume_min)); maximum=D(str(volume_max))
    loss_per_lot=(distance/tick)*tick_value
    raw=budget/loss_per_lot
    steps=int((raw/step).to_integral_value(rounding=ROUND_FLOOR))
    max_steps=int((maximum/step).to_integral_value(rounding=ROUND_FLOOR))
    volume=step*min(steps,max_steps)
    if volume<minimum:
        return SizingResult(False,0.0,float(budget),0.0,float(distance),
          "Minimum broker volume would exceed the risk budget")
    estimated_loss=volume*loss_per_lot
    if estimated_loss>budget:
        return SizingResult(False,0.0,float(budget),float(estimated_loss),float(distance),
          "Rounded volume breaches the risk budget")
    return SizingResult(True,float(volume),float(budget),float(estimated_loss),
      float(distance),"Approved within configured risk limit")
