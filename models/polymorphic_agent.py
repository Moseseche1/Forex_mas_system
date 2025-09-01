import numpy as np
import random
import logging
from datetime import datetime
from typing import Dict, List, Any
from services.genetic_evolver import GeneticEvolver
from services.drl_adaptor import DeepAdaptiveLearner
from config.settings import settings

logger = logging.getLogger(__name__)

class PolymorphicAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.genetic_evolver = GeneticEvolver()
        self.drl_adaptor = DeepAdaptiveLearner(state_size=10, action_size=5)
        self.strategy_dna = self._initialize_dna()
        self.learning_rate = 0.01
        self.adaptation_counter = 0
        
    def _initialize_dna(self) -> Dict:
        """Initialize agent's strategy DNA"""
        return {
            'risk_appetite': random.uniform(0.1, 0.9),
            'timeframe_preference': random.choice(['M1', 'M5', 'M15', 'H1', 'H4']),
            'indicator_weight_rsi': random.uniform(0.1, 1.0),
            'indicator_weight_macd': random.uniform(0.1, 1.0),
            'indicator_weight_bb': random.uniform(0.1, 1.0),
            'volume_sensitivity': random.uniform(0.1, 1.0),
            'volatility_tolerance': random.uniform(0.1, 1.0),
            'trend_bias': random.uniform(-1.0, 1.0),
            'max_position_hold_time': random.randint(1, 24),
            'news_sensitivity': random.uniform(0.1, 1.0)
        }
    
    async def adapt_strategy(self, market_data: Dict, performance: Dict) -> Dict:
        """Adapt strategy based on market conditions and performance"""
        self.adaptation_counter += 1
        
        # Multi-level adaptation
        if self.adaptation_counter % 10 == 0:
            await self._genetic_adaptation(performance)
        
        if self.adaptation_counter % 5 == 0:
            await self._reinforcement_learning_adaptation(market_data, performance)
        
        # Continuous parameter adjustment
        self._continuous_parameter_optimization(market_data)
        
        return self.strategy_dna
    
    async def _genetic_adaptation(self, performance: Dict):
        """Genetic algorithm adaptation"""
        try:
            # Evolve based on performance
            evolved_strategies = await self.genetic_evolver.evolve_strategies(
                {self.agent_id: performance}
            )
            
            if evolved_strategies:
                # Adopt best evolved strategy
                self.strategy_dna = self._blend_dna(
                    self.strategy_dna, 
                    evolved_strategies[0],
                    blend_factor=0.3
                )
                
        except Exception as e:
            logger.error(f"Genetic adaptation failed: {e}")
    
    async def _reinforcement_learning_adaptation(self, market_data: Dict, performance: Dict):
        """Deep reinforcement learning adaptation"""
        try:
            # Convert market data to state representation
            state = self._market_data_to_state(market_data)
            
            # Get action from DRL model
            action = self.drl_adaptor.act(state)
            
            # Calculate reward based on performance
            reward = self._calculate_reward(performance)
            
            # Get next state (simulated)
            next_state = self._simulate_next_state(state, action)
            
            # Remember experience
            self.drl_adaptor.remember(state, action, reward, next_state, False)
            
            # Replay and learn
            self.drl_adaptor.replay()
            
        except Exception as e:
            logger.error(f"RL adaptation failed: {e}")
    
    def _continuous_parameter_optimization(self, market_data: Dict):
        """Continuous parameter adjustment based on market conditions"""
        # Adjust parameters based on volatility
        volatility = market_data.get('volatility', 0.1)
        self.strategy_dna['risk_appetite'] *= (1 - volatility * 0.1)
        
        # Adjust based on trend strength
        trend_strength = market_data.get('trend_strength', 0)
        self.strategy_dna['trend_bias'] += trend_strength * 0.01
        
        # Ensure parameters stay within bounds
        self._normalize_dna_parameters()
    
    def _blend_dna(self, dna1: Dict, dna2: Dict, blend_factor: float = 0.5) -> Dict:
        """Blend two DNA sets"""
        blended = {}
        
        for key in set(dna1.keys()) | set(dna2.keys()):
            if key in dna1 and key in dna2:
                if isinstance(dna1[key], (int, float)):
                    blended[key] = blend_factor * dna1[key] + (1 - blend_factor) * dna2[key]
                else:
                    blended[key] = dna1[key] if random.random() < 0.5 else dna2[key]
            else:
                blended[key] = dna1.get(key, dna2.get(key))
        
        return blended
    
    def _normalize_dna_parameters(self):
        """Ensure DNA parameters stay within valid ranges"""
        for key, value in self.strategy_dna.items():
            if isinstance(value, float):
                self.strategy_dna[key] = max(0.0, min(1.0, value))
