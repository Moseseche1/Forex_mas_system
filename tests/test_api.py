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

def test_paper_order_is_sized_and_never_sent_to_broker(monkeypatch, tmp_path):
    from core.paper_ledger import PaperLedger
    import main
    monkeypatch.setattr(main, "paper_ledger", PaperLedger(str(tmp_path/"paper.sqlite3")))
    payload={"symbol":"EURUSD","side":"long","equity":1000,"day_start_equity":1000,
      "risk_fraction":.01,"entry_price":1.1,"stop_price":1.099,"target_price":1.102,
      "tick_size":.0001,"tick_value_per_lot":10,"volume_min":.01,"volume_max":100,
      "volume_step":.01,"estimated_cost_per_lot":1,"commission_per_lot":7}
    response=client.post("/paper/orders",json=payload)
    assert response.status_code==200
    data=response.json()
    assert data["broker_order_sent"] is False
    assert data["position"]["status"]=="open"
    assert data["position"]["volume_lots"] <= 1.0
    position_id=data["position"]["id"]
    closed=client.post(f"/paper/positions/{position_id}/close",json={"exit_price":1.101})
    assert closed.status_code==200
    assert closed.json()["broker_order_sent"] is False
    assert closed.json()["position"]["status"]=="closed"
    assert client.get("/paper/positions?status=closed").json()["positions"][0]["id"]==position_id

def test_paper_order_rejects_over_budget_minimum_lot(monkeypatch, tmp_path):
    from core.paper_ledger import PaperLedger
    import main
    monkeypatch.setattr(main, "paper_ledger", PaperLedger(str(tmp_path/"paper.sqlite3")))
    payload={"symbol":"EURUSD","side":"long","equity":10,"day_start_equity":10,
      "risk_fraction":.01,"entry_price":1.1,"stop_price":1.09,
      "tick_size":.0001,"tick_value_per_lot":10,"volume_min":.01,"volume_max":100,
      "volume_step":.01,"estimated_cost_per_lot":0}
    assert client.post("/paper/orders",json=payload).status_code==409

def test_paper_daily_loss_gate(monkeypatch, tmp_path):
    from core.paper_ledger import PaperLedger
    import main
    monkeypatch.setattr(main, "paper_ledger", PaperLedger(str(tmp_path/"paper.sqlite3")))
    now=__import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    start=now.replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
    main.paper_ledger.create_position(symbol="EURUSD",side="long",volume_lots=1,
      entry_price=1.1,stop_price=1.099,target_price=None,tick_size=.0001,
      tick_value_per_lot=10,commission_per_lot=0,risk_budget=1,estimated_loss=1)
    pos=main.paper_ledger.list_positions("open")[0]
    main.paper_ledger.close_position(pos["id"],1.096)
    assert main.paper_ledger.realized_pnl_since(start) < -3
    payload={"symbol":"EURUSD","side":"long","equity":1000,"day_start_equity":1000,
      "risk_fraction":.01,"entry_price":1.1,"stop_price":1.099,
      "tick_size":.0001,"tick_value_per_lot":10,"volume_min":.01,"volume_max":100,
      "volume_step":.01,"estimated_cost_per_lot":0}
    assert client.post("/paper/orders",json=payload).status_code==409
