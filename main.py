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

# Polymorphic Architecture Imports
from config.settings import settings
from services.market_data import market_data_service
from services.broker_service import mt5_broker
from services.quantum_service import quantum_service
from services.genetic_evolver import GeneticEvolver
from services.drl_adaptor import DeepAdaptiveLearner
from services.self_learning_analyzer import SelfLearningMarketAnalyzer
from services.evolution_orchestrator import EvolutionOrchestrator
from models.polymorphic_agent import PolymorphicAgent
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

class PolymorphicTradingSystem:
    def __init__(self):
        self.agents = [PolymorphicAgent(i) for i in range(settings.TOTAL_AGENTS)]
        self.evolution_orchestrator = EvolutionOrchestrator()
        self.market_analyzer = SelfLearningMarketAnalyzer()
        self.genetic_evolver = GeneticEvolver()
        self.is_running = False
        self.market_data = {}
        self.performance_metrics = {}
        self.evolution_cycle = 0
        self.last_evolution = datetime.utcnow()
        
    async def initialize_system(self):
        """Initialize polymorphic trading system"""
        logger.info("Initializing Polymorphic Trading System...")
        
        # Initialize MT5 connection
        if mt5_broker.initialize_mt5():
            logger.info("MT5 broker initialized successfully")
        
        # Initialize market data service
        await market_data_service.init_session()
        logger.info("Market data service initialized")
        
        # Initialize learning systems
        await self._initialize_learning_systems()
        
        logger.info("Polymorphic Trading System initialization complete")
    
    async def _initialize_learning_systems(self):
        """Initialize machine learning components"""
        try:
            # Load historical data for initial training
            historical_data = await self._load_historical_data()
            
            # Pre-train market analyzer
            if historical_data:
                await self.market_analyzer.analyze_market_regime(historical_data)
            
            logger.info("Learning systems initialized with historical data")
            
        except Exception as e:
            logger.error(f"Learning system initialization failed: {e}")
    
    async def trading_cycle(self):
        """Main trading loop with polymorphic adaptation"""
        while self.is_running:
            try:
                # Update market data
                self.market_data = await self._update_market_data()
                
                # Analyze market regime
                market_analysis = await self.market_analyzer.analyze_market_regime(self.market_data)
                
                # Execute polymorphic trading
                for agent in self.agents:
                    if agent.active:
                        # Adapt strategy based on market conditions
                        adapted_strategy = await agent.adapt_strategy(
                            self.market_data, 
                            self.performance_metrics.get(agent.agent_id, {})
                        )
                        
                        # Make trading decision with adapted strategy
                        decision = await self._make_trading_decision(agent, adapted_strategy)
                        
                        # Execute trade if decision is valid
                        if decision.get('action') != 'HOLD':
                            trade_result = await self._execute_trade(decision)
                            decision['trade_result'] = trade_result
                        
                        # Log decision for learning
                        await self._log_trading_decision(agent.agent_id, decision, adapted_strategy)
                
                # Autonomous evolution
                await self._execute_autonomous_evolution()
                
                logger.info(f"Trading cycle completed. Evolution cycle: {self.evolution_cycle}")
                await asyncio.sleep(settings.UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                await asyncio.sleep(10)
    
    async def _make_trading_decision(self, agent, strategy: Dict) -> Dict:
        """Make trading decision using polymorphic strategy"""
        try:
            # Get quantum-enhanced signal
            quantum_signal = quantum_service.generate_quantum_trading_signal(
                'EURUSD',  # Default symbol, can be made configurable
                self.market_data
            )
            
            # Apply strategy DNA to decision making
            decision_confidence = (
                strategy.get('risk_appetite', 0.5) * 
                quantum_signal.get('confidence', 0.5) *
                (1 - self.market_data.get('volatility', 0.1))
            )
            
            # Determine action based on strategy and market conditions
            if quantum_signal.get('signal', 0) > strategy.get('trend_bias', 0) and decision_confidence > 0.4:
                action = 'BUY'
            elif quantum_signal.get('signal', 0) < strategy.get('trend_bias', 0) and decision_confidence > 0.4:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # Calculate position size based on risk appetite
            position_size = self._calculate_position_size(
                strategy.get('risk_appetite', 0.5),
                self.market_data.get('volatility', 0.1)
            )
            
            return {
                'action': action,
                'amount': position_size,
                'confidence': decision_confidence,
                'quantum_signal': quantum_signal.get('signal', 0),
                'strategy_dna': strategy,
                'market_regime': self.market_data.get('regime', 'unknown'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trading decision failed: {e}")
            return {'action': 'HOLD', 'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def _execute_trade(self, decision: Dict) -> Dict:
        """Execute trade through MT5 broker"""
        try:
            if mt5_broker.connected:
                # Calculate stop loss and take profit based on volatility
                volatility = self.market_data.get('volatility', 0.01)
                stop_loss_pips = max(15, int(volatility * 1000))
                take_profit_pips = stop_loss_pips * 2
                
                trade_result = mt5_broker.execute_trade(
                    symbol='EURUSD',  # Configurable
                    trade_type=decision['action'],
                    volume=decision['amount'],
                    stop_loss=stop_loss_pips,
                    take_profit=take_profit_pips
                )
                
                return trade_result
            else:
                return {'success': False, 'error': 'MT5 not connected', 'simulated': True}
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _log_trading_decision(self, agent_id: int, decision: Dict, strategy: Dict):
        """Log trading decision for learning and analysis"""
        try:
            log_entry = {
                'agent_id': agent_id,
                'decision': decision,
                'strategy_dna': strategy,
                'market_data': self.market_data,
                'timestamp': datetime.utcnow(),
                'evolution_cycle': self.evolution_cycle
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
                'win_rate': 0
            }
        
        metrics = self.performance_metrics[agent_id]
        metrics['total_trades'] += 1
        
        # Update win rate (simulated for now)
        if decision.get('action') != 'HOLD':
            is_win = random.random() > 0.4  # Simulated win/loss
            if is_win:
                metrics['winning_trades'] += 1
                metrics['total_profit'] += random.uniform(0.1, 0.5)
            else:
                metrics['losing_trades'] += 1
                metrics['total_profit'] -= random.uniform(0.1, 0.3)
            
            metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])
    
    async def _execute_autonomous_evolution(self):
        """Execute autonomous evolution cycles"""
        try:
            current_time = datetime.utcnow()
            
            # Continuous evolution every 10 cycles
            if self.evolution_cycle % 10 == 0:
                system_state = {
                    'market_data': self.market_data,
                    'performance_metrics': self.performance_metrics,
                    'agent_count': len(self.agents),
                    'system_uptime': (current_time - self.last_evolution).total_seconds(),
                    'evolution_cycle': self.evolution_cycle
                }
                
                await self.evolution_orchestrator.orchestrate_evolution(system_state)
                
                # Genetic evolution every 100 cycles
                if self.evolution_cycle % 100 == 0:
                    await self._execute_genetic_evolution()
                
                self.last_evolution = current_time
            
            self.evolution_cycle += 1
            
        except Exception as e:
            logger.error(f"Autonomous evolution failed: {e}")
    
    async def _execute_genetic_evolution(self):
        """Execute genetic evolution of strategies"""
        try:
            if self.performance_metrics:
                evolved_strategies = await self.genetic_evolver.evolve_strategies(self.performance_metrics)
                
                # Update agents with evolved strategies
                for agent_id, strategy in evolved_strategies.items():
                    if agent_id < len(self.agents):
                        self.agents[agent_id].strategy_dna = strategy
                
                logger.info(f"Genetic evolution completed: {len(evolved_strategies)} strategies evolved")
                
        except Exception as e:
            logger.error(f"Genetic evolution failed: {e}")
    
    async def _update_market_data(self) -> Dict:
        """Update market data from various sources"""
        try:
            # Get real market data
            realtime_data = await market_data_service.get_real_time_data('EURUSD')
            technicals = await market_data_service.get_technical_indicators('EURUSD')
            
            market_data = {
                'realtime': realtime_data,
                'technicals': technicals,
                'volatility': technicals.get('volatility', 0.1) if technicals else 0.1,
                'trend_strength': technicals.get('trend_strength', 0) if technicals else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in Redis for dashboard
            if redis_client:
                redis_client.set('market_data', json.dumps(market_data))
                redis_client.set('performance_metrics', json.dumps(self.performance_metrics))
                redis_client.set('evolution_cycle', self.evolution_cycle)
                
            return market_data
            
        except Exception as e:
            logger.error(f"Market data update error: {e}")
            return {
                'volatility': 0.1, 
                'trend_strength': 0,
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def _calculate_position_size(self, risk_appetite: float, volatility: float) -> float:
        """Calculate position size based on risk and volatility"""
        base_size = 0.01  # Minimum lot size
        risk_adjusted = base_size * (1 + risk_appetite * 2)  # 0.01 to 0.03 lots
        volatility_adjusted = risk_adjusted * (1 - min(volatility, 0.5))  # Reduce size in high volatility
        return round(volatility_adjusted, 2)
    
    async def _load_historical_data(self) -> Dict:
        """Load historical data for initial training"""
        # This would typically load from database or API
        # For now, return simulated data
        return {
            'volatility': 0.1,
            'trend_strength': 0.2,
            'volume_ratio': 1.0,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Clean shutdown of polymorphic system"""
        self.is_running = False
        await market_data_service.close_session()
        mt5_broker.shutdown_mt5()
        
        # Save learning state
        await self._save_learning_state()
        
        logger.info("Polymorphic Trading System shutdown complete")
    
    async def _save_learning_state(self):
        """Save learning state for future sessions"""
        try:
            learning_state = {
                'evolution_cycle': self.evolution_cycle,
                'performance_metrics': self.performance_metrics,
                'market_patterns': self.market_analyzer.learned_patterns,
                'last_evolution': self.last_evolution.isoformat(),
                'save_time': datetime.utcnow().isoformat()
            }
            
            if redis_client:
                redis_client.set('learning_state', json.dumps(learning_state))
                
            logger.info("Learning state saved successfully")
            
        except Exception as e:
            logger.error(f"Learning state save failed: {e}")

# Initialize polymorphic system
trading_system = PolymorphicTradingSystem()

# Create FastAPI app with dashboard
app = FastAPI(
    title="Polymorphic Forex AI Trading System", 
    version="3.0",
    description="Self-evolving trading system that adapts to broker algorithms",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include dashboard routes
app.include_router(dashboard_router)

@app.get("/")
async def root():
    return {
        "message": "Polymorphic Forex AI Trading System is running",
        "version": "3.0",
        "status": "operational" if trading_system.is_running else "stopped",
        "features": [
            "polymorphic_architecture",
            "self_evolution",
            "genetic_algorithm",
            "deep_reinforcement_learning",
            "quantum_enhanced",
            "mt5_integration",
            "autonomous_adaptation"
        ],
        "evolution_cycle": trading_system.evolution_cycle
    }

@app.get("/status")
async def get_status():
    active_agents = sum(1 for agent in trading_system.agents if agent.active)
    
    system_info = {
        "active_agents": active_agents,
        "total_agents": len(trading_system.agents),
        "system_running": trading_system.is_running,
        "mt5_connected": mt5_broker.connected,
        "quantum_available": quantum_service.quantum_available,
        "evolution_cycle": trading_system.evolution_cycle,
        "last_evolution": trading_system.last_evolution.isoformat(),
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
async def start_system():
    if not trading_system.is_running:
        await trading_system.initialize_system()
        trading_system.is_running = True
        asyncio.create_task(trading_system.trading_cycle())
        return {
            "status": "started", 
            "message": "Polymorphic trading system started",
            "evolution_cycle": trading_system.evolution_cycle
        }
    return {
        "status": "already_running", 
        "message": "System already running",
        "evolution_cycle": trading_system.evolution_cycle
    }

@app.post("/stop")
async def stop_system():
    await trading_system.shutdown()
    return {
        "status": "stopped", 
        "message": "Polymorphic trading system stopped",
        "evolution_cycle": trading_system.evolution_cycle
    }

@app.post("/evolve")
async def trigger_evolution():
    """Manually trigger evolution cycle"""
    try:
        system_state = {
            'market_data': trading_system.market_data,
            'performance_metrics': trading_system.performance_metrics,
            'agent_count': len(trading_system.agents),
            'evolution_cycle': trading_system.evolution_cycle
        }
        
        await trading_system.evolution_orchestrator.orchestrate_evolution(system_state)
        trading_system.evolution_cycle += 1
        
        return {
            "status": "evolution_triggered",
            "message": "Manual evolution cycle completed",
            "new_evolution_cycle": trading_system.evolution_cycle
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evolution failed: {str(e)}")

@app.post("/kill")
async def emergency_stop(auth_token: str):
    if auth_token != settings.KILL_SWITCH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    await trading_system.shutdown()
    for agent in trading_system.agents:
        agent.active = False
    
    logger.critical("EMERGENCY STOP ACTIVATED - ALL TRADING HALTED")
    return {
        "status": "emergency_stop", 
        "message": "All trading activities stopped",
        "evolution_cycle": trading_system.evolution_cycle
    }

@app.get("/agent/{agent_id}")
async def get_agent_status(agent_id: int):
    if 0 <= agent_id < len(trading_system.agents):
        agent = trading_system.agents[age
