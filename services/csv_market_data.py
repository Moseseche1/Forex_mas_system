"""Validated CSV historical OHLC loader; never returns fabricated prices."""
import csv
from math import isfinite
from pathlib import Path
REQUIRED=("timestamp","open","high","low","close")
def load_ohlc_csv(path:str|Path)->list[dict]:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(str(p))
    out=[]
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
            raise ValueError("CSV requires timestamp, open, high, low, close columns")
        for line,row in enumerate(reader,start=2):
            try: item={"timestamp":row["timestamp"],**{k:float(row[k]) for k in REQUIRED[1:]}}
            except (TypeError,ValueError): raise ValueError(f"Invalid OHLC value on line {line}")
            if not item["timestamp"]: raise ValueError(f"Missing timestamp on line {line}")
            o,h,l,c=(item[k] for k in REQUIRED[1:])
            if not all(isfinite(x) and x>0 for x in (o,h,l,c)) or h<max(o,c) or l>min(o,c) or h<l: raise ValueError(f"Invalid OHLC range on line {line}")
            out.append(item)
    if not out: raise ValueError("CSV contains no price bars")
    return out
