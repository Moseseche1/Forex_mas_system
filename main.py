"""Olympus FX API — safe research/paper-trading control plane.

This service deliberately does not submit broker orders. Broker execution must
be implemented as a separately tested service behind an independent risk gate.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from config.settings import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("olympus_fx")

AGENT_ROLES = [
    "executive", "market_intelligence", "macro_research",
    "quantitative_research", "strategy", "risk", "portfolio",
]

class Runtime:
    def __init__(self) -> None:
        self.running = False
        self.started_at: str | None = None
        self.cycles_completed = 0
        self.last_error: str | None = None

runtime = Runtime()
app = FastAPI(
    title="Olympus FX",
    version="0.1.0",
    description="Multi-agent forex research and paper-trading control plane. Live execution is disabled.",
)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _require_admin(token: str | None) -> None:
    expected = settings.KILL_SWITCH_TOKEN
    if not expected or not token or token != expected:
        raise HTTPException(status_code=401, detail="Valid X-Admin-Token required")

@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "Olympus FX",
        "status": "running" if runtime.running else "stopped",
        "mode": "paper",
        "live_trading_enabled": False,
        "timestamp": _now(),
    }

@app.get("/health")
@app.get("/system/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "api": "healthy", "timestamp": _now()}

@app.get("/status")
async def status() -> dict[str, Any]:
    return {
        "system_running": runtime.running,
        "trading_mode": "paper",
        "live_trading_enabled": False,
        "total_agents": len(AGENT_ROLES),
        "active_agents": len(AGENT_ROLES) if runtime.running else 0,
        "agents": [
            {"agent_id": i, "role": role, "active": runtime.running,
             "state": "research_only", "orders_submitted": 0}
            for i, role in enumerate(AGENT_ROLES)
        ],
        "cycles_completed": runtime.cycles_completed,
        "started_at": runtime.started_at,
        "last_error": runtime.last_error,
        "timestamp": _now(),
    }

@app.get("/agent/{agent_id}")
async def get_agent(agent_id: int) -> dict[str, Any]:
    if agent_id < 0 or agent_id >= len(AGENT_ROLES):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {
        "agent_id": agent_id,
        "strategy": AGENT_ROLES[agent_id],
        "active": runtime.running,
        "state": "research_only",
        "orders_submitted": 0,
    }

@app.post("/start")
async def start_system() -> dict[str, str]:
    runtime.running = True
    runtime.started_at = runtime.started_at or _now()
    logger.info("Research/paper-trading control plane started; live orders remain disabled")
    return {"status": "started", "mode": "paper", "live_trading_enabled": "false"}

@app.post("/stop")
async def stop_system() -> dict[str, str]:
    runtime.running = False
    logger.info("System stopped")
    return {"status": "stopped"}

@app.post("/kill")
async def emergency_stop(x_admin_token: str | None = Header(default=None)) -> dict[str, str]:
    _require_admin(x_admin_token)
    runtime.running = False
    logger.critical("Emergency stop activated")
    return {"status": "emergency_stop", "live_trading_enabled": "false"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Olympus FX</title><style>
    body{font:16px system-ui;background:#0b1220;color:#e7edf7;margin:0;padding:32px}
    main{max-width:900px;margin:auto}section{background:#172338;padding:24px;border-radius:16px}
    .pill{display:inline-block;padding:6px 12px;border-radius:20px;background:#244b43}
    </style></head><body><main><h1>OLYMPUS FX</h1>
    <p>Multi-agent forex research platform</p><section>
    <span class="pill">PAPER MODE ONLY</span><h2>Execution safeguard</h2>
    <p>Live broker order submission is disabled. No trades are sent from this service.</p>
    <p>API health: <a href="/health">/health</a> · System status: <a href="/status">/status</a> · API docs: <a href="/docs">/docs</a></p>
    </section></main></body></html>"""

@app.on_event("startup")
async def startup_event() -> None:
    # Intentionally do not auto-start trading or connect to a broker at import/startup.
    logger.info("Olympus FX API ready in paper-only mode")

@app.on_event("shutdown")
async def shutdown_event() -> None:
    runtime.running = False
