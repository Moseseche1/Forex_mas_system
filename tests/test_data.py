import pytest
from services.csv_market_data import load_ohlc_csv
def test_csv_loader(tmp_path):
 p=tmp_path/"bars.csv"; p.write_text("timestamp,open,high,low,close\n2025-01-01,1,1.2,.9,1.1\n")
 assert load_ohlc_csv(p)[0]["close"]==1.1
def test_csv_loader_rejects_bad_values(tmp_path):
 p=tmp_path/"bad.csv"; p.write_text("timestamp,open,high,low,close\n2025-01-01,nope,1.2,.9,1.1\n")
 with pytest.raises(ValueError): load_ohlc_csv(p)
