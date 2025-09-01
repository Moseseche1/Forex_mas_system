import pytest
from fastapi.testclient import TestClient
from main import app
from config.settings import settings

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint returns successful response"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_status_endpoint():
    """Test status endpoint returns system information"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "active_agents" in data
    assert "total_agents" in data
    assert "system_running" in data

def test_kill_switch_authentication():
    """Test kill switch requires valid authentication"""
    response = client.post("/kill", params={"auth_token": "invalid_token"})
    assert response.status_code == 401

def test_agent_endpoint():
    """Test agent endpoint returns agent information"""
    response = client.get("/agent/0")
    assert response.status_code == 200
    data = response.json()
    assert "agent_id" in data
    assert "active" in data
    assert "strategy" in data

def test_nonexistent_agent():
    """Test non-existent agent returns 404"""
    response = client.get("/agent/999")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_system_start_stop():
    """Test system start and stop functionality"""
    # Start system
    start_response = client.post("/start")
    assert start_response.status_code == 200
    
    # Stop system
    stop_response = client.post("/stop")
    assert stop_response.status_code == 200

def test_dashboard_endpoint():
    """Test dashboard endpoint returns HTML"""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
