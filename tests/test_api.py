from fastapi.testclient import TestClient
from main import app, runtime
from config.settings import settings
client=TestClient(app)
def setup_function():
    runtime.running=False; runtime.started_at=None
def test_health_and_paper_safety():
    assert client.get("/health").json()["status"]=="ok"
    assert client.get("/").json()["live_trading_enabled"] is False
def test_status_and_agent():
    data=client.get("/status").json()
    assert data["total_agents"]==7 and data["active_agents"]==0 and data["execution_workers_active"] is False
    assert all(a["orders_submitted"]==0 and not a["active"] for a in data["agents"])
    assert client.get("/agent/0").status_code==200 and client.get("/agent/999").status_code==404
def test_start_stop_and_kill_switch(monkeypatch):
    assert client.post("/start").json()["mode"]=="paper" and runtime.running
    assert client.post("/stop").json()["status"]=="stopped"
    monkeypatch.setattr(settings,"KILL_SWITCH_TOKEN","test-secret")
    assert client.post("/kill").status_code==401
    assert client.post("/kill",headers={"X-Admin-Token":"test-secret"}).status_code==200
def test_dashboard():
    r=client.get("/dashboard")
    assert r.status_code==200 and "PAPER MODE ONLY" in r.text
def test_risk_api():
    r=client.post("/risk/position-size",json={"equity":100,"risk_fraction":.01,"entry_price":1.1,"stop_price":1.099,"tick_size":.0001,"tick_value_per_lot":10,"volume_min":.01,"volume_max":100,"volume_step":.01,"estimated_cost_per_lot":0})
    assert r.status_code==200 and r.json()["approved"] and r.json()["estimated_loss"]<=1
def test_backtest_api():
    bars=[{"open":1.1,"high":1.101,"low":1.099,"close":1.1},{"open":1.1,"high":1.1045,"low":1.0995,"close":1.104},{"open":1.104,"high":1.105,"low":1.103,"close":1.104}]
    r=client.post("/backtest",json={"bars":bars,"signals":[1,0,0]})
    assert r.status_code==200 and r.json()["trade_count"]==1 and r.json()["assumptions"]["live_orders_sent"] is False
def test_market_data_fails_closed_without_credentials(monkeypatch):
    monkeypatch.setattr(settings,"OANDA_API_KEY","")
    monkeypatch.setattr(settings,"OANDA_ACCOUNT_ID","")
    assert client.get("/market/status").json()["synthetic_fallback"] is False
    assert client.get("/market/quote/EURUSD").status_code==503

def test_signal_api():
    r=client.post("/signals/ema-crossover",json={"closes":[1,1,1,1,2,3,4,5],"fast":2,"slow":4})
    assert r.status_code==200 and len(r.json()["signals"])==8

def test_multi_agent_analysis_and_risk_veto():
    rising=[1.0+i*.001 for i in range(40)]
    r=client.post("/agents/analyze",json={"closes":rising,"risk_fraction":.01,"current_drawdown":0})
    assert r.status_code==200
    assert r.json()["execution"]["live_orders_sent"] is False
    blocked=client.post("/agents/analyze",json={"closes":rising,"risk_fraction":.01,"current_drawdown":.04})
    assert blocked.status_code==200 and blocked.json()["decision"]=="no_trade"
