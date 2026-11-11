"""Explicit placeholder: no quantum hardware or validated quantum model is connected."""
from datetime import datetime,timezone
from typing import Dict

class QuantumTradingService:
    def __init__(self):
        self.quantum_available=False
    def generate_quantum_random(self)->float:
        raise RuntimeError("Quantum randomness is not configured; no pseudo-quantum substitute is provided")
    def generate_quantum_trading_signal(self,symbol:str,market_data:Dict)->Dict:
        return {"symbol":symbol,"signal":0,"confidence":0,"available":False,
          "decision":"no_trade","reason":"No validated quantum model is configured",
          "timestamp":datetime.now(timezone.utc).isoformat(),"live_orders_sent":False}

quantum_service=QuantumTradingService()
