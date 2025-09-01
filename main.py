from fastapi import FastAPI, HTTPException
import logging
import asyncio
from datetime import datetime
from config.settings import settings
from services.broker_service import mt5_broker
from services.market_data import market_data_service
from services.quantum_service import quantum_service
from models.polymorphic_agent import PolymorphicAgent

# Setup logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self):
        self.agents = [PolymorphicAgent(i) for i in range(settings.TOTAL_AGENTS)]
        self.is_running = False
        
    async def initialize_system(self):
        logger.info("Initializing Trading System...")
        await market_data_service.init_session()
        
        if mt5_broker.initialize_mt5():
            logger.info("MT5 initialized successfully")
        
        logger.info("Trading System initialization complete")
    
    async def trading_cycle(self):
        while self.is_running:
            try:
                for agent in self.agents:
                    if agent.should_trade():
                        market_data = await market_data_service.get_real_time_data('EURUSD')
                        strategy = await agent.adapt_strategy(market_data, {})
                        
                        logger.info(f"Agent {agent.agent_id} trading with strategy: {strategy}")
                
                await asyncio.sleep(settings.UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                await asyncio.sleep(10)
    
    async def shutdown(self):
        self.is_running = False
        await market_data_service.close_session()
        mt5_broker.shutdown_mt5()

trading_system = TradingSystem()
app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Polymorphic Forex AI Trading System",
        "status": "running" if trading_system.is_running else "stopped",
        "agents": len(trading_system.agents)
    }

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

@app.get("/status")
async def get_status():
    active_agents = sum(1 for agent in trading_system.agents if agent.active)
    return {
        "active_agents": active_agents,
        "total_agents": len(trading_system.agents),
        "system_running": trading_system.is_running
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
