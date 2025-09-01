import numpy as np
import random
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
from config.settings import settings

logger = logging.getLogger(__name__)

class GeneticEvolver:
    def __init__(self):
        self.population_size = 50
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.strategy_pool = []
        self.generation = 0
        
    async def evolve_strategies(self, performance_data: Dict) -> List[Dict]:
        """Evolve trading strategies based on performance"""
        # Evaluate current strategies
        fitness_scores = self._calculate_fitness(performance_data)
        
        # Selection - keep top performers
        selected = self._tournament_selection(fitness_scores)
        
        # Crossover - create new strategies
        offspring = self._crossover(selected)
        
        # Mutation - introduce randomness
        mutated = self._mutate(offspring)
        
        # Update strategy pool
        self.strategy_pool = self._create_new_generation(selected, mutated)
        self.generation += 1
        
        logger.info(f"Generation {self.generation} evolved: {len(self.strategy_pool)} strategies")
        return self.strategy_pool
    
    def _calculate_fitness(self, performance_data: Dict) -> Dict[int, float]:
        """Calculate fitness scores based on performance metrics"""
        fitness_scores = {}
        
        for agent_id, metrics in performance_data.items():
            # Multi-objective fitness function
            sharpe = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0)
            drawdown = metrics.get('max_drawdown', 1)
            profit_factor = metrics.get('profit_factor', 0)
            
            # Fitness formula (weighted components)
            fitness = (sharpe * 0.3 + 
                      win_rate * 0.2 + 
                      profit_factor * 0.3 + 
                      (1 - min(drawdown, 0.5)) * 0.2)
            
            fitness_scores[agent_id] = max(0, fitness)
        
        return fitness_scores
    
    def _tournament_selection(self, fitness_scores: Dict, tournament_size: int = 3) -> List[int]:
        """Tournament selection for genetic algorithm"""
        selected = []
        agent_ids = list(fitness_scores.keys())
        
        while len(selected) < self.population_size // 2:
            # Random tournament
            tournament = random.sample(agent_ids, min(tournament_size, len(agent_ids)))
            winner = max(tournament, key=lambda x: fitness_scores.get(x, 0))
            selected.append(winner)
        
        return selected
    
    def _crossover(self, selected_agents: List[int]) -> List[Dict]:
        """Crossover strategies to create offspring"""
        offspring = []
        
        for i in range(0, len(selected_agents), 2):
            if i + 1 < len(selected_agents):
                parent1 = self._get_agent_strategy(selected_agents[i])
                parent2 = self._get_agent_strategy(selected_agents[i + 1])
                
                if random.random() < self.crossover_rate:
                    child_strategy = self._blend_strategies(parent1, parent2)
                    offspring.append(child_strategy)
        
        return offspring
    
    def _mutate(self, strategies: List[Dict]) -> List[Dict]:
        """Mutate strategies to introduce novelty"""
        mutated = []
        
        for strategy in strategies:
            if random.random() < self.mutation_rate:
                mutated_strategy = self._apply_mutation(strategy)
                mutated.append(mutated_strategy)
            else:
                mutated.append(strategy)
        
        return mutated
    
    def _blend_strategies(self, strategy1: Dict, strategy2: Dict) -> Dict:
        """Blend two strategies using weighted average"""
        blended = {}
        
        for key in set(strategy1.keys()) | set(strategy2.keys()):
            if key in strategy1 and key in strategy2:
                # Blend numerical parameters
                if isinstance(strategy1[key], (int, float)):
                    weight = random.random()
                    blended[key] = weight * strategy1[key] + (1 - weight) * strategy2[key]
                else:
                    # Random choice for non-numerical
                    blended[key] = random.choice([strategy1[key], strategy2[key]])
            else:
                blended[key] = strategy1.get(key, strategy2.get(key))
        
        return blended
    
    def _apply_mutation(self, strategy: Dict) -> Dict:
        """Apply random mutations to strategy parameters"""
        mutated = strategy.copy()
        
        for key in mutated:
            if isinstance(mutated[key], (int, float)):
                # Gaussian mutation for numerical parameters
                mutation_strength = random.choice([0.1, 0.2, 0.5])
                mutated[key] *= random.gauss(1, mutation_strength)
                
                # Occasionally introduce completely new value
                if random.random() < 0.05:
                    mutated[key] = random.uniform(0, 1)
        
        return mutated
