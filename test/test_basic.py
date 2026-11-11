from fastapi.testclient import TestClient
from main import app, runtime
client=TestClient(app)
def setup_function():
    runtime.running=False
def test_root_endpoint():
    r=client.get("/")
    assert r.status_code==200 and r.json()["mode"]=="paper"
def test_status_endpoint():
    d=client.get("/status").json()
    assert "active_agents" in d and "total_agents" in d and "system_running" in d
def test_kill_requires_authentication():
    assert client.post("/kill",params={"auth_token":"invalid"}).status_code==401
def test_agent_endpoint():
    d=client.get("/agent/0").json()
    assert "agent_id" in d and "active" in d and "strategy" in d
def test_nonexistent_agent():
    assert client.get("/agent/999").status_code==404
def test_start_stop():
    assert client.post("/start").status_code==200
    assert client.post("/stop").status_code==200
def test_dashboard():
    assert client.get("/dashboard").status_code==200
