"""Olympus FX API — research and paper-trading only; no broker orders."""
import logging
import hmac
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from dataclasses import asdict
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from config.settings import settings
from core.risk import size_position
from core.backtest import run_backtest
from core.signals import ema_crossover, rsi
from services.market_data import market_data_service, MarketDataUnavailable
from agents.engine import analyze_market
from core.paper_ledger import PaperLedger

logging.basicConfig(level=settings.LOG_LEVEL)
logger=logging.getLogger("olympus_fx")
AGENT_ROLES=["executive","market_intelligence","quantitative_research","market_regime","volatility","risk","portfolio"]

class Runtime:
    def __init__(self):
        self.running=False; self.started_at=None; self.cycles_completed=0; self.last_error=None
runtime=Runtime()
paper_ledger=PaperLedger(settings.PAPER_LEDGER_PATH)
@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("Olympus FX ready in paper-only mode")
    yield
    runtime.running=False
app=FastAPI(title="Olympus FX",version="0.2.0",description="Research and paper-trading API. Live broker execution is disabled.",lifespan=lifespan)

def _now(): return datetime.now(timezone.utc).isoformat()
def _require_admin(token):
    expected=settings.KILL_SWITCH_TOKEN
    if not expected or not token or not hmac.compare_digest(token,expected): raise HTTPException(status_code=401,detail="Valid X-Admin-Token required")

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
    estimated_cost_per_lot:float=Field(ge=0)

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
      "total_agents":len(AGENT_ROLES),"active_agents":0,"configured_agents":len(AGENT_ROLES),"execution_workers_active":False,
      "agents":[{"agent_id":i,"role":role,"active":False,"state":"on_demand_research_only","orders_submitted":0} for i,role in enumerate(AGENT_ROLES)],
      "cycles_completed":runtime.cycles_completed,"started_at":runtime.started_at,"last_error":runtime.last_error,"timestamp":_now()}
@app.get("/agent/{agent_id}")
async def get_agent(agent_id:int):
    if agent_id<0 or agent_id>=len(AGENT_ROLES): raise HTTPException(status_code=404,detail="Agent not found")
    return {"agent_id":agent_id,"strategy":AGENT_ROLES[agent_id],"active":False,"state":"on_demand_research_only","orders_submitted":0}
@app.post("/start")
async def start_system():
    runtime.running=True; runtime.started_at=runtime.started_at or _now()
    return {"status":"research_runtime_enabled","mode":"paper","agents_started":False,"live_trading_enabled":False}
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
class AgentAnalysisRequest(BaseModel):
    closes:list[float]=Field(min_length=30,max_length=100000)
    risk_fraction:float=Field(default=.01,gt=0,le=.01)
    current_drawdown:float=Field(default=0,ge=0,lt=1)
    max_bar_volatility:float=Field(default=.01,gt=0)

@app.post("/agents/analyze")
async def analyze_agents(request:AgentAnalysisRequest):
    try: return analyze_market(**request.model_dump())
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

class PaperOrderRequest(BaseModel):
    symbol: str = Field(min_length=6, max_length=6, pattern=r"^[A-Za-z]{6}$")
    side: str
    equity: float = Field(gt=0)
    day_start_equity: float = Field(gt=0)
    risk_fraction: float = Field(gt=0, le=.01)
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    target_price: float | None = Field(default=None, gt=0)
    tick_size: float = Field(gt=0)
    tick_value_per_lot: float = Field(gt=0)
    volume_min: float = Field(gt=0)
    volume_max: float = Field(gt=0)
    volume_step: float = Field(gt=0)
    estimated_cost_per_lot: float = Field(ge=0)
    commission_per_lot: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_direction(self):
        if self.side not in ("long", "short"):
            raise ValueError("side must be long or short")
        if self.side == "long" and self.stop_price >= self.entry_price:
            raise ValueError("Long stop must be below entry")
        if self.side == "short" and self.stop_price <= self.entry_price:
            raise ValueError("Short stop must be above entry")
        if self.target_price is not None:
            if self.side == "long" and self.target_price <= self.entry_price:
                raise ValueError("Long target must be above entry")
            if self.side == "short" and self.target_price >= self.entry_price:
                raise ValueError("Short target must be below entry")
        return self

class PaperCloseRequest(BaseModel):
    exit_price: float = Field(gt=0)

@app.post("/paper/orders")
async def create_paper_order(request: PaperOrderRequest, x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    day_equity = paper_ledger.get_or_set_day_start_equity(day_start[:10], request.day_start_equity)
    realized_today = paper_ledger.realized_pnl_since(day_start)
    daily_loss_limit = day_equity * settings.MAX_DRAWDOWN
    if realized_today <= -daily_loss_limit:
        raise HTTPException(status_code=409, detail="Daily paper loss limit reached; new positions are blocked")
    try:
        sizing = size_position(
            equity=request.equity,
            risk_fraction=request.risk_fraction,
            entry_price=request.entry_price,
            stop_price=request.stop_price,
            tick_size=request.tick_size,
            tick_value_per_lot=request.tick_value_per_lot,
            volume_min=request.volume_min,
            volume_max=request.volume_max,
            volume_step=request.volume_step,
            estimated_cost_per_lot=request.estimated_cost_per_lot,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not sizing.approved:
        raise HTTPException(status_code=409, detail=sizing.reason)
    position = paper_ledger.create_position(
        symbol=request.symbol.upper(), side=request.side,
        volume_lots=sizing.volume_lots, entry_price=request.entry_price,
        stop_price=request.stop_price, target_price=request.target_price,
        tick_size=request.tick_size, tick_value_per_lot=request.tick_value_per_lot,
        commission_per_lot=request.commission_per_lot,
        risk_budget=sizing.risk_budget, estimated_loss=sizing.estimated_loss,
    )
    return {"mode":"paper","broker_order_sent":False,"day_start_equity":day_equity,"position":position,"sizing":asdict(sizing)}

@app.get("/paper/positions")
async def list_paper_positions(status: str | None = None, x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    if status not in (None, "open", "closed"):
        raise HTTPException(status_code=422, detail="status must be open or closed")
    return {"mode":"paper","broker_orders_sent":0,"positions":paper_ledger.list_positions(status)}

@app.post("/paper/positions/{position_id}/close")
async def close_paper_position(position_id: str, request: PaperCloseRequest, x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    try:
        position = paper_ledger.close_position(position_id, request.exit_price)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if position is None:
        raise HTTPException(status_code=404, detail="Paper position not found")
    return {"mode":"paper","broker_order_sent":False,"position":position}

@app.get("/paper/risk")
async def paper_risk_status(day_start_equity: float = Query(gt=0), x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    now=datetime.now(timezone.utc)
    day_start=now.replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
    day_equity=paper_ledger.get_or_set_day_start_equity(day_start[:10], day_start_equity)
    realized=paper_ledger.realized_pnl_since(day_start)
    return {"mode":"paper","realized_pnl_today":realized,"day_start_equity":day_equity,
            "daily_loss_limit":day_equity*settings.MAX_DRAWDOWN,
            "new_orders_allowed":realized > -(day_equity*settings.MAX_DRAWDOWN),
            "live_trading_enabled":False}

@app.get("/dashboard",response_class=HTMLResponse)
async def dashboard():
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Olympus FX</title><style>body{font:16px system-ui;background:#0b1220;color:#e7edf7;margin:0;padding:32px}main{max-width:900px;margin:auto}section{background:#172338;padding:24px;border-radius:16px}.pill{display:inline-block;padding:6px 12px;border-radius:20px;background:#244b43}a{color:#9cc7ff}</style></head><body><main><h1>OLYMPUS FX</h1><p>Multi-agent forex research platform</p><section><span class="pill">PAPER MODE ONLY</span><h2>Execution safeguard</h2><p>Live broker order submission is disabled. No trades are sent from this service.</p><p><a href="/health">Health</a> · <a href="/status">Status</a> · <a href="/docs">API docs</a></p></section></main></body></html>"""
