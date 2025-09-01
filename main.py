from fastapi import FastAPI, HTTPException, BackgroundTasks
import redis
import pymongo
import time
import numpy as np
import random
import logging
import asyncio
from datetime import datetime
import os
import json

# New imports for upgrades
from config.settings import settings
from services.market_data import market_data_service
from services.broker_service import mt5_broker
from services.quantum_service import quantum_service
from dashboard import router as dashboard_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize connections
try:
    redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
    mongo_client = pymongo.MongoClient(settings.MONGO_URI)
    db = mongo_client['forex_mas']
    logger.info("Database connections established")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    # Fallback to in-memory operation
    redis_client = None
    db = None

class TradingAgent:
    def __init__(self, agent_id: int, strategy: str):
        self.agent_id = agent_id
        self.strategy = strategy
        self.active = True
        self.balance = 10000.0
        self.emotions = {'confidence': 0.5, 'caution': 0.3, 'aggression': 0.2}
        self.performance = {'wins': 0, 'losses': 0, 'sharpe': 0.0}
        self.quantum_signals = []
        
    def update_emotions(self, market_volatility: float, profit_loss: float):
        # Emotional intelligence logic
        self.emotions['confidence'] = max(0.1, min(0.9, 
            self.emotions['confidence'] + profit_loss * 0.1))
        self.emotions['caution'] = max(0.1, min(0.9,
            self.emotions['caution'] + market_volatility * 0.2))
        self.emotions['aggression'] = max(0.1, min(0.9,
            self.emotions['confidence'] - self.emotions['caution']))
        
        # Save to Redis if available
        if redis_client:
            redis_client.set(f'agent_{self.agent_id}_emotions', json.dumps(self.emotions))
    
    async def make_trading_decision(self) -> dict:
        """Enhanced trading decision with quantum and market data"""
        try:
            # Get market data
            market_data = await market_data_service.get_technical_indicators('EURUSD')
            quantum_signal = quantum_service.generate_quantum_trading_signal('EURUSD', market_data)
            
            self.quantum_signals.append(quantum_signal)
            if len(self.quantum_signals) > 100:
                self.quantum_signals = self.quantum_signals[-100:]
            
            # Decision logic with quantum influence
            if quantum_signal['signal'] > 0.3 and self.emotions['confidence'] > 0.4:
                action = 'BUY'
            elif quantum_signal['signal'] < -0.3 and self.emotions['confidence'] > 0.4:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            decision = {
                'action': action,
                'amount': random.uniform(0.1, 2.0) * (1 + self.emotions['aggression']),
                'confidence': self.emotions['confidence'],
                'quantum_signal': quantum_signal['signal'],
                'quantum_confidence': quantum_signal['confidence'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Execute trade if not HOLD
            if action != 'HOLD' and mt5_broker.connected:
                trade_result = mt5_broker.execute_trade(
                    symbol='EURUSD',
                    trade_type=action,
                    volume=decision['amount'],
                    stop_loss=0.02,  # 2% stop loss
                    take_profit=0.04  # 4% take profit
                )
                decision['trade_result'] = trade_result
            
            return decision
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} decision error: {e}")
            return {'action': 'HOLD', 'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

class AITradingSystem:
    def __init__(self):
        self.agents = [TradingAgent(i, settings.AGENT_STRATEGIES[i % 5]) for i in range(settings.TOTAL_AGENTS)]
        self.is_running = False
        self.market_data = {}
        
    async def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing trading system...")
        
        # Initialize MT5 connection
        if mt5_broker.initialize_mt5():
            logger.info("MT5 broker initialized successfully")
        
        # Initialize market data service
        await market_data_service.init_session()
        logger.info("Market data service initialized")
        
        logger.info("Trading system initialization complete")
        
    async def trading_cycle(self):
        """Main trading loop with enhanced features"""
        while self.is_running:
            try:
                # Update market data
                self.market_data = await self._update_market_data()
                
                for agent in self.agents:
                    if agent.active:
                        # Get live market volatility
                        volatility = self.market_data.get('volatility', 0.1)
                        
                        # Simulate P&L for emotion update
                        profit_loss = random.uniform(-0.01, 0.02)
                        agent.update_emotions(volatility, profit_loss)
                        
                        # Make trading decision with quantum enhancement
                        decision = await agent.make_trading_decision()
                        
                        # Log decision
                        if db:
                            db['trading_decisions'].insert_one({
                                'agent_id': agent.agent_id,
                                'decision': decision,
                                'timestamp': datetime.utcnow(),
                                'emotions': agent.emotions
                            })
                
                logger.info(f"Trading cycle completed at {datetime.utcnow()}")
                await asyncio.sleep(settings.UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                await asyncio.sleep(10)
    
    async def _update_market_data(self) -> dict:
        """Update market data from various sources"""
        try:
            # Get real market data
            realtime_data = await market_data_service.get_real_time_data('EURUSD')
            technicals = await market_data_service.get_technical_indicators('EURUSD')
            
            market_data = {
                'realtime': realtime_data,
                'technicals': technicals,
                'volatility': technicals.get('volatility', 0.1) if technicals else 0.1,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in Redis for dashboard
            if redis_client:
                redis_client.set('market_data', json.dumps(market_data))
                
            return market_data
            
        except Exception as e:
            logger.error(f"Market data update error: {e}")
            return {'volatility': 0.1, 'timestamp': datetime.utcnow().isoformat()}
    
    async def shutdown(self):
        """Clean shutdown of system components"""
        self.is_running = False
        await market_data_service.close_session()
        mt5_broker.shutdown_mt5()
        logger.info("Trading system shutdown complete")

# Initialize system
trading_system = AITradingSystem()

# Create FastAPI app with dashboard
app = FastAPI(title="Forex AI Trading System", version="2.0")
app.include_router(dashboard_router)

@app.get("/")
async def root():
    return {
        "message": "Forex AI Trading System is running",
        "version": "2.0",
        "status": "operational" if trading_system.is_running else "stopped",
        "features": ["quantum_enhanced", "real_market_data", "mt5_integration", "web_dashboard"]
    }

@app.get("/status")
async def get_status():
    active_agents = sum(1 for agent in trading_system.agents if agent.active)
    
    # Get system info
    system_info = {
        "active_agents": active_agents,
        "total_agents": len(trading_system.agents),
        "system_running": trading_system.is_running,
        "mt5_connected": mt5_broker.connected,
        "quantum_available": quantum_service.quantum_available,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add agent details
    agents_details = {}
    for agent in trading_system.agents:
        agents_details[f'agent_{agent.agent_id}'] = {
            'active': agent.active,
            'strategy': agent.strategy,
            'emotions': agent.emotions,
            'balance': agent.balance,
            'performance': agent.performance
        }
    
    system_info['agents'] = agents_details
    return system_info

@app.post("/start")
async def start_system():
    if not trading_system.is_running:
        await trading_system.initialize_system()
        trading_system.is_running = True
        asyncio.create_task(trading_system.trading_cycle())
        return {"status": "started", "message": "Trading system started"}
    return {"status": "already_running", "message": "System already running"}

@app.post("/stop")
async def stop_system():
    await trading_system.shutdown()
    return {"status": "stopped", "message": "Trading system stopped"}

@app.post("/kill")
async def emergency_stop(auth_token: str):
    if auth_token != settings.KILL_SWITCH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    await trading_system.shutdown()
    for agent in trading_system.agents:
        agent.active = False
    
    logger.critical("EMERGENCY STOP ACTIVATED - ALL TRADING HALTED")
    return {"status": "emergency_stop", "message": "All trading activities stopped"}

@app.get("/agent/{agent_id}")
async def get_agent_status(agent_id: int):
    if 0 <= agent_id < len(trading_system.agents):
        agent = trading_system.agents[agent_id]
        return {
            "agent_id": agent.agent_id,
            "active": agent.active,
            "strategy": agent.strategy,
            "emotions": agent.emotions,
            "balance": agent.balance,
            "performance": agent.performance,
            "quantum_signals_count": len(agent.quantum_signals)
        }
    raise HTTPException(status_code=404, detail="Agent not found")

@app.get("/market/data")
async def get_market_data():
    """Get current market data"""
    market_data = trading_system.market_data or {}
    return {
        "market_data": market_data,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/system/health")
async def system_health():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "redis": redis_client is not None,
            "mongodb": db is not None,
            "mt5": mt5_broker.connected,
            "quantum": quantum_service.quantum_available,
            "market_data": market_data_service.session is not None
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Startup event - initialize system"""
    logger.info("Forex AI Trading System starting up...")
    await trading_system.initialize_system()
    
    # Auto-start the system in development
    if not settings.is_production:
        trading_system.is_running = True
        asyncio.create_task(trading_system.trading_cycle())
        logger.info("System auto-started in development mode")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - clean up resources"""
    logger.info("Shutting down Forex AI Trading System...")
    await trading_system.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
