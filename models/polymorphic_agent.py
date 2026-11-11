"""Deterministic research agent; never executes or authorizes broker orders."""
from typing import Dict

class PolymorphicAgent:
    def __init__(self,agent_id:int):
        self.agent_id=agent_id
        self.active=False
        self.strategy_dna={"risk_appetite":0.01,"aggression":0.0}
        self.adaptation_counter=0
    async def adapt_strategy(self,market_data:Dict,performance:Dict)->Dict:
        volatility=float(market_data.get("volatility",0.0))
        if volatility<0: raise ValueError("volatility cannot be negative")
        # Deterministic research parameter adjustment only; never raises risk above 1%.
        self.strategy_dna["risk_appetite"]=min(.01,max(.001,.01/(1+volatility*10)))
        self.strategy_dna["aggression"]=0.0
        self.adaptation_counter+=1
        return dict(self.strategy_dna)
    def should_trade(self)->bool:
        return False
