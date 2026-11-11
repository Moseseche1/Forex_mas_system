"""Olympus FX API — research and paper-trading only; no broker orders."""
import logging
from datetime import datetime, timezone
from typing import Any
from dataclasses import asdict
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from config.settings import settings
from core.risk import size_position
from core.backtest import run_backtest
from core.signals import ema_crossover, rsi
from services.market_data import market_data_service, MarketDataUnavailable

logging.basicConfig(level=settings.LOG_LEVEL)
logger=logging.getLogger("olympus_fx")
AGENT_ROLES=["executive","market_intelligence","macro_research","quantitative_research","strategy","risk","portfolio"]

class Runtime:
    def __init__(self):
        self.running=False; self.started_at=None; self.cycles_completed=0; self.last_error=None
runtime=Runtime()
app=FastAPI(title="Olympus FX",version="0.2.0",description="Research and paper-trading API. Live broker execution is disabled.")

def _now(): return datetime.now(timezone.utc).isoformat()
def _require_admin(token):
    expected=settings.KILL_SWITCH_TOKEN
    if not expected or not token or token!=expected: raise HTTPException(status_code=401,detail="Valid X-Admin-Token required")

class RiskRequest(BaseModel):
    equity:float=Field(gt=0)
    risk_fraction:float=Field(gt=0,le=.01)
    entry_price:float=Field(gt=0)
    stop_price:float=Field(gt=0)
    tick_size:float=Field(gt=0)
    tick_value_per_lot:float=Field(gt=0)
    volume_min:float=Field(gt=0)
    volume_max:float=Field(gt=0)
    volume_step:float=Field(gt=0)

class Bar(BaseModel):
    open:float=Field(gt=0); high:float=Field(gt=0); low:float=Field(gt=0); close:float=Field(gt=0)
    @model_validator(mode="after")
    def validate_range(self):
        if self.high<max(self.open,self.close) or self.low>min(self.open,self.close) or self.high<self.low:
            raise ValueError("Invalid OHLC range")
        return self

class BacktestRequest(BaseModel):
    bars:list[Bar]=Field(min_length=3,max_length=100000)
    signals:list[int]=Field(min_length=3,max_length=100000)
    initial_equity:float=Field(default=1000,gt=0)
    risk_fraction:float=Field(default=.01,gt=0,le=.01)
    stop_distance:float=Field(default=.002,gt=0)
    reward_risk:float=Field(default=2,gt=0)
    tick_size:float=Field(default=.0001,gt=0)
    tick_value_per_lot:float=Field(default=10,gt=0)
    volume_min:float=Field(default=.01,gt=0)
    volume_max:float=Field(default=100,gt=0)
    volume_step:float=Field(default=.01,gt=0)
    spread:float=Field(default=.0001,ge=0)
    slippage:float=Field(default=.00002,ge=0)
    commission_per_lot:float=Field(default=7,ge=0)
    @model_validator(mode="after")
    def validate_lengths_and_signals(self):
        if len(self.bars)!=len(self.signals): raise ValueError("bars and signals must have equal lengths")
        if any(s not in (-1,0,1) for s in self.signals): raise ValueError("signals must be -1, 0, or 1")
        return self

class SignalRequest(BaseModel):
    closes:list[float]=Field(min_length=2,max_length=100000)
    fast:int=Field(default=12,ge=1)
    slow:int=Field(default=26,ge=2)
    @model_validator(mode="after")
    def periods(self):
        if self.fast>=self.slow: raise ValueError("fast must be less than slow")
        return self

@app.get("/")
async def root()->dict[str,Any]:
    return {"name":"Olympus FX","status":"running" if runtime.running else "stopped","mode":"paper","live_trading_enabled":False,"timestamp":_now()}
@app.get("/health")
@app.get("/system/health")
async def health(): return {"status":"ok","api":"healthy","timestamp":_now()}
@app.get("/status")
async def status():
    return {"system_running":runtime.running,"trading_mode":"paper","live_trading_enabled":False,
      "total_agents":len(AGENT_ROLES),"active_agents":len(AGENT_ROLES) if runtime.running else 0,
      "agents":[{"agent_id":i,"role":role,"active":runtime.running,"state":"research_only","orders_submitted":0} for i,role in enumerate(AGENT_ROLES)],
      "cycles_completed":runtime.cycles_completed,"started_at":runtime.started_at,"last_error":runtime.last_error,"timestamp":_now()}
@app.get("/agent/{agent_id}")
async def get_agent(agent_id:int):
    if agent_id<0 or agent_id>=len(AGENT_ROLES): raise HTTPException(status_code=404,detail="Agent not found")
    return {"agent_id":agent_id,"strategy":AGENT_ROLES[agent_id],"active":runtime.running,"state":"research_only","orders_submitted":0}
@app.post("/start")
async def start_system():
    runtime.running=True; runtime.started_at=runtime.started_at or _now()
    return {"status":"started","mode":"paper","live_trading_enabled":False}
@app.post("/stop")
async def stop_system():
    runtime.running=False
    return {"status":"stopped"}
@app.post("/kill")
async def emergency_stop(x_admin_token:str|None=Header(default=None)):
    _require_admin(x_admin_token); runtime.running=False
    logger.critical("Emergency stop activated")
    return {"status":"emergency_stop","live_trading_enabled":False}
@app.post("/risk/position-size")
async def risk_position_size(request:RiskRequest):
    try: return asdict(size_position(**request.model_dump()))
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc))
@app.post("/backtest")
async def backtest(request:BacktestRequest):
    try:
        payload=request.model_dump()
        bars=payload.pop("bars")
        signals=payload.pop("signals")
        return run_backtest(bars,signals,**payload)
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc))
@app.post("/signals/ema-crossover")
async def ema_signal(request:SignalRequest):
    try: return {"signals":ema_crossover(request.closes,request.fast,request.slow),"rsi_14":rsi(request.closes,14),"live_orders_sent":False}
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc))
@app.get("/market/status")
async def market_status():
    return market_data_service.get_service_status()
@app.get("/market/quote/{symbol}")
async def market_quote(symbol:str):
    try: return await market_data_service.get_real_time_data(symbol)
    except MarketDataUnavailable as exc: raise HTTPException(status_code=503,detail=str(exc))
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc))
    except Exception as exc:
        logger.exception("Market data provider request failed")
        raise HTTPException(status_code=502,detail="Market data provider unavailable")
@app.get("/market/candles/{symbol}")
async def market_candles(symbol:str,granularity:str="M5",count:int=500):
    try: return {"candles":await market_data_service.get_historical_data(symbol,granularity,count),"source":"oanda","read_only":True}
    except MarketDataUnavailable as exc: raise HTTPException(status_code=503,detail=str(exc))
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc))
    except Exception:
        logger.exception("Historical market data request failed")
        raise HTTPException(status_code=502,detail="Market data provider unavailable")
@app.get("/dashboard",response_class=HTMLResponse)
async def dashboard():
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Olympus FX</title><style>body{font:16px system-ui;background:#0b1220;color:#e7edf7;margin:0;padding:32px}main{max-width:900px;margin:auto}section{background:#172338;padding:24px;border-radius:16px}.pill{display:inline-block;padding:6px 12px;border-radius:20px;background:#244b43}a{color:#9cc7ff}</style></head><body><main><h1>OLYMPUS FX</h1><p>Multi-agent forex research platform</p><section><span class="pill">PAPER MODE ONLY</span><h2>Execution safeguard</h2><p>Live broker order submission is disabled. No trades are sent from this service.</p><p><a href="/health">Health</a> · <a href="/status">Status</a> · <a href="/docs">API docs</a></p></section></main></body></html>"""
@app.on_event("startup")
async def startup_event(): logger.info("Olympus FX ready in paper-only mode")
@app.on_event("shutdown")
async def shutdown_event(): runtime.running=False
