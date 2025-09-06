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
from typing import Dict, List, Optional, Any

# Import configurations and services
from config.settings import settings
from services.market_data import market_data_service
from services.broker_service import mt5_broker
from services.quantum_service import quantum_service

# Import error handling utilities
from utils.error_handler import error_handler, retry_with_backoff, CircuitOpenError

# Import models
from models.polymorphic_agent import PolymorphicAgent

# Import dashboard
from dashboard import router as dashboard_router

# Setup logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_mas.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize connections with error handling
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST, 
        port=settings.REDIS_PORT, 
        decode_responses=True,
        socket_connect_timeout=5,
        retry_on_timeout=True
    )
    # Test Redis connection
    redis_client.ping()
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

try:
    mongo_client = pymongo.MongoClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000
    )
    # Test MongoDB connection
    mongo_client.admin.command('ping')
    db = mongo_client['forex_mas']
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    db = None
    mongo_client = None

class PolymorphicTradingSystem:
    def __init__(self):
        self.agents = [PolymorphicAgent(i) for i in range(settings.TOTAL_AGENTS)]
        self.is_running = False
        self.market_data = {}
        self.performance_metrics = {}
        self.system_start_time = None
        self.trading_cycles_completed = 0
        self.last_health_check = datetime.utcnow()
        
    @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=10.0)
    async def initialize_system(self):
        """Initialize trading system with retry logic"""
        logger.info("Initializing Polymorphic Trading System...")
        self.system_start_time = datetime.utcnow()
        
        try:
            # Initialize market data service with error handling
            await error_handler.execute_with_circuit_breaker(
                "market_data_init",
                market_data_service.init_session
            )
            
            # Initialize MT5 connection with error handling
            mt5_initialized = await error_handler.execute_with_circuit_breaker(
                "mt5_init",
                mt5_broker.initialize_mt5
            )
            
            if mt5_initialized:
                logger.info("MT5 broker initialized successfully")
            else:
                logger.warning("MT5 running in simulation mode")
            
            logger.info("Trading System initialization complete")
            
        except CircuitOpenError:
            logger.error("Circuit breaker open during system initialization")
            raise
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0, max_delay=5.0)
    async def trading_cycle(self):
        """Main trading loop with enhanced error handling"""
        while self.is_running:
            try:
                # Update market data with circuit breaker protection
                self.market_data = await error_handler.execute_with_circuit_breaker(
                    "market_data_update",
                    self._update_market_data
                )
                
                # Execute trading for all agents
                await self._execute_agent_trading_cycle()
                
                # Perform system health check
                await self._perform_health_check()
                
                self.trading_cycles_completed += 1
                logger.info(f"Trading cycle {self.trading_cycles_completed} completed successfully")
                
                await asyncio.sleep(settings.UPDATE_INTERVAL)
                
            except CircuitOpenError as e:
                logger.warning(f"Circuit breaker open: {e}. Pausing trading temporarily.")
                await asyncio.sleep(30)  # Longer pause for circuit breaker
                
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                await asyncio.sleep(10)  # Shorter pause for other errors
    
    async def _execute_agent_trading_cycle(self):
        """Execute trading for all agents with individual error handling"""
        trading_tasks = []
        
        for agent in self.agents:
            if agent.active:
                # Create task for each agent with error handling
                task = self._execute_agent_trading(agent)
                trading_tasks.append(task)
        
        # Execute all agent tasks concurrently
        if trading_tasks:
            await asyncio.gather(*trading_tasks, return_exceptions=True)
    
    async def _execute_agent_trading(self, agent: PolymorphicAgent):
        """Execute trading for a single agent with comprehensive error handling"""
        try:
            # Get market data with timeout protection
            market_data = await error_handler.execute_with_timeout(
                market_data_service.get_real_time_data,
                timeout=5.0,
                symbol='EURUSD'
            )
            
            if not market_data:
                logger.warning(f"No market data for agent {agent.agent_id}")
                return
            
            # Update emotions with error handling
            await error_handler.execute_with_circuit_breaker(
                f"agent_{agent.agent_id}_emotions",
                agent.update_emotions,
                market_data.get('volatility', 0.1),
                random.uniform(-0.01, 0.02)
            )
            
            # Make trading decision with circuit breaker
            decision = await error_handler.execute_with_circuit_breaker(
                f"agent_{agent.agent_id}_decision",
                agent.make_trading_decision
            )
            
            if decision and decision.get('action') != 'HOLD':
                # Execute trade with circuit breaker protection
                trade_result = await error_handler.execute_with_circuit_breaker(
                    "mt5_trading",
                    self._execute_trade,
                    decision
                )
                decision['trade_result'] = trade_result
            
            # Log decision with error handling
            await error_handler.execute_with_circuit_breaker(
                "database_logging",
                self._log_trading_decision,
                agent.agent_id,
                decision,
                agent.strategy_dna
            )
            
        except CircuitOpenError:
            logger.warning(f"Circuit open for agent {agent.agent_id}, skipping cycle")
        except Exception as e:
            logger.error(f"Agent {agent.agent_id} trading error: {e}")
    
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    async def _execute_trade(self, decision: Dict) -> Dict:
        """Execute a trade with retry logic"""
        try:
            if mt5_broker.connected:
                trade_result = mt5_broker.execute_trade(
                    symbol='EURUSD',
                    trade_type=decision['action'],
                    volume=decision['amount'],
                    stop_loss=0.02,
                    take_profit=0.04
                )
                return trade_result
            else:
                return {
                    'success': False, 
                    'error': 'MT5 not connected', 
                    'simulated': True,
                    'decision': decision
                }
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    async def _update_market_data(self) -> Dict:
        """Update market data with retry logic"""
        try:
            realtime_data = await market_data_service.get_real_time_data('EURUSD')
            technicals = await market_data_service.get_technical_indicators('EURUSD')
            
            market_data = {
                'realtime': realtime_data,
                'technicals': technicals,
                'volatility': technicals.get('volatility', 0.1) if technicals else 0.1,
                'trend_strength': technicals.get('trend_strength', 0) if technicals else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in Redis if available
            if redis_client:
                try:
                    redis_client.setex(
                        'market_data', 
                        60,  # 1 minute expiration
                        json.dumps(market_data)
                    )
                except Exception as e:
                    logger.warning(f"Redis store failed: {e}")
                    
            return market_data
            
        except Exception as e:
            logger.error(f"Market data update error: {e}")
            return {
                'volatility': 0.1, 
                'trend_strength': 0,
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'source': 'fallback'
            }
    
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    async def _log_trading_decision(self, agent_id: int, decision: Dict, strategy: Dict):
        """Log trading decision with retry logic"""
        try:
            log_entry = {
                'agent_id': agent_id,
                'decision': decision,
                'strategy_dna': strategy,
                'market_data': self.market_data,
                'timestamp': datetime.utcnow(),
                'trading_cycle': self.trading_cycles_completed
            }
            
            # Store in database if available
            if db:
                db['trading_decisions'].insert_one(log_entry)
            
            # Update performance metrics
            self._update_performance_metrics(agent_id, decision)
            
        except Exception as e:
            logger.error(f"Decision logging failed: {e}")
    
    def _update_performance_metrics(self, agent_id: int, decision: Dict):
        """Update performance metrics for adaptation"""
        if agent_id not in self.performance_metrics:
            self.performance_metrics[agent_id] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0.5
            }
        
        metrics = self.performance_metrics[agent_id]
        metrics['total_trades'] += 1
        
        # Simulated performance tracking (replace with real P&L)
        if decision.get('action') != 'HOLD':
            is_win = random.random() > 0.4  # Simulated win/loss
            if is_win:
                metrics['winning_trades'] += 1
                metrics['total_profit'] += random.uniform(0.1, 0.5)
            else:
                metrics['losing_trades'] += 1
                metrics['total_profit'] -= random.uniform(0.1, 0.3)
            
            metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])
    
    async def _perform_health_check(self):
        """Perform system health check"""
        current_time = datetime.utcnow()
        if (current_time - self.last_health_check).total_seconds() > 300:  # Every 5 minutes
            try:
                health_status = await self.get_system_health()
                logger.info(f"System health check: {health_status}")
                self.last_health_check = current_time
            except Exception as e:
                logger.error(f"Health check failed: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            'system_running': self.is_running,
            'agents_active': sum(1 for agent in self.agents if agent.active),
            'trading_cycles_completed': self.trading_cycles_completed,
            'system_uptime': (datetime.utcnow() - self.system_start_time).total_seconds() if self.system_start_time else 0,
            'redis_connected': redis_client is not None,
            'mongodb_connected': db is not None,
            'mt5_connected': mt5_broker.connected,
            'market_data_available': bool(self.market_data),
            'circuit_breaker_status': error_handler.get_all_status(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0, max_delay=5.0)
    async def shutdown(self):
        """Clean shutdown with retry logic"""
        self.is_running = False
        
        try:
            # Close market data session
            await error_handler.execute_with_circuit_breaker(
                "market_data_shutdown",
                market_data_service.close_session
            )
            
            # Shutdown MT5
            await error_handler.execute_with_circuit_breaker(
                "mt5_shutdown",
                mt5_broker.shutdown_mt5
            )
            
            logger.info("Polymorphic Trading System shutdown complete")
            
        except Exception as e:
            logger.error(f"System shutdown error: {e}")
            raise

# Initialize polymorphic system
trading_system = PolymorphicTradingSystem()

# Create FastAPI app with dashboard
app = FastAPI(
    title="Polymorphic Forex AI Trading System", 
    version="3.0",
    description="Self-evolving trading system with enhanced error handling",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include dashboard routes
app.include_router(dashboard_router)

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Polymorphic Forex AI Trading System",
        "version": "3.0",
        "status": "operational" if trading_system.is_running else "stopped",
        "agents_active": sum(1 for agent in trading_system.agents if agent.active),
        "features": [
            "polymorphic_architecture",
            "enhanced_error_handling",
            "circuit_breaker_pattern",
            "automatic_retries",
            "quantum_enhanced",
            "mt5_integration"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    active_agents = sum(1 for agent in trading_system.agents if agent.active)
    
    system_info = {
        "active_agents": active_agents,
        "total_agents": len(trading_system.agents),
        "system_running": trading_system.is_running,
        "mt5_connected": mt5_broker.connected,
        "trading_cycles_completed": trading_system.trading_cycles_completed,
        "system_uptime": (datetime.utcnow() - trading_system.system_start_time).total_seconds() if trading_system.system_start_time else 0,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add agent details
    agents_details = {}
    for agent in trading_system.agents:
        agents_details[f'agent_{agent.agent_id}'] = {
            'active': agent.active,
            'strategy_type': agent.strategy_dna.get('strategy_type', 'adaptive'),
            'risk_appetite': agent.strategy_dna.get('risk_appetite', 0.5),
            'adaptation_count': agent.adaptation_counter,
            'performance': trading_system.performance_metrics.get(agent.agent_id, {})
        }
    
    system_info['agents'] = agents_details
    return system_info

@app.post("/start")
async def start_system(background_tasks: BackgroundTasks):
    """Start the trading system"""
    if not trading_system.is_running:
        try:
            await trading_system.initialize_system()
            trading_system.is_running = True
            background_tasks.add_task(trading_system.trading_cycle)
            return {
                "status": "started", 
                "message": "Trading system started successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"System startup failed: {str(e)}")
    return {
        "status": "already_running", 
        "message": "System already running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/stop")
async def stop_system():
    """Stop the trading system"""
    try:
        await trading_system.shutdown()
        return {
            "status": "stopped", 
            "message": "Trading system stopped successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System shutdown failed: {str(e)}")

@app.get("/system/health")
async def system_health():
    """Get comprehensive system health information"""
    try:
        health_status = await trading_system.get_system_health()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/system/circuit-status")
async def get_circuit_status():
    """Get status of all circuit breakers"""
    try:
        return error_handler.get_all_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Circuit status check failed: {str(e)}")

@app.post("/system/reset-circuit/{circuit_name}")
async def reset_circuit(circuit_name: str, auth_token: str):
    """Reset a specific circuit breaker"""
    if auth_token != settings.KILL_SWITCH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    try:
        error_handler.reset_circuit(circuit_name)
        return {
            "status": "success", 
            "message": f"Circuit {circuit_name} reset successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Circuit reset failed: {str(e)}")

@app.post("/system/reset-all-circuits")
async def reset_all_circuits(auth_token: str):
    """Reset all circuit breakers"""
    if auth_token != settings.KILL_SWITCH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    try:
        error_handler.reset_all_circuits()
        return {
            "status": "success", 
            "message": "All circuits reset successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Circuit reset failed: {str(e)}")

@app.post("/kill")
async def emergency_stop(auth_token: str):
    """Emergency stop the system"""
    if auth_token != settings.KILL_SWITCH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    try:
        trading_system.is_running = False
        for agent in trading_system.agents:
            agent.active = False
        
        
        # Force shutdown components
        await market_data_service.close_session()
        mt5_broker.shutdown_mt5()
        
        logger.critical("EMERGENCY STOP ACTIVATED - ALL TRADING HALTED")
        return {
            "status": "emergency_stop", 
            "message": "All trading activities stopped immediately",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Startup event - initialize system"""
    logger.info("Polymorphic Forex AI Trading System starting up...")
    
    # Auto-start the system in development
    if not settings.is_production:
        try:
            await trading_system.initialize_system()
            trading_system.is_running = True
            # Start trading cycle in background
            asyncio.create_task(trading_system.trading_cycle())
            logger.info("System auto-started in development mode")
        except Exception as e:
            logger.error(f"Auto-start failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - clean up resources"""
    logger.info("Shutting down Polymorphic Forex AI Trading System...")
    try:
        await trading_system.shutdown()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```
