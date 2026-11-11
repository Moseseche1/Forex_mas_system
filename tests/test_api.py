from fastapi.testclient import TestClient
from main import app, runtime
from config.settings import settings

client = TestClient(app)

def setup_function():
    runtime.running = False
    runtime.started_at = None

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_root_never_enables_live_trading():
    data = client.get("/").json()
    assert data["mode"] == "paper"
    assert data["live_trading_enabled"] is False

def test_status_reports_research_only_agents():
    data = client.get("/status").json()
    assert data["total_agents"] == 7
    assert all(a["orders_submitted"] == 0 for a in data["agents"])

def test_agent_endpoint_and_missing_agent():
    assert client.get("/agent/0").status_code == 200
    assert client.get("/agent/999").status_code == 404

def test_start_stop_are_paper_only():
    assert client.post("/start").json()["mode"] == "paper"
    assert runtime.running is True
    assert client.post("/stop").json()["status"] == "stopped"
    assert runtime.running is False

def test_kill_switch_requires_token(monkeypatch):
    monkeypatch.setattr(settings, "KILL_SWITCH_TOKEN", "test-secret")
    assert client.post("/kill").status_code == 401
    response = client.post("/kill", headers={"X-Admin-Token": "test-secret"})
    assert response.status_code == 200
    assert response.json()["status"] == "emergency_stop"

def test_dashboard_serves_html():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
