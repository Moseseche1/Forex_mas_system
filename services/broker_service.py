"""Disabled legacy MT5 execution adapter.

This compatibility shim intentionally never initializes MT5 and never sends orders.
A separately reviewed execution worker is a future milestone, not part of this API.
"""
import logging
logger=logging.getLogger(__name__)

class MT5TradingService:
    def __init__(self):
        self.connected=False
        self.enabled=False
    def initialize_mt5(self)->bool:
        logger.warning("MT5 broker connection is disabled in this release")
        return False
    def shutdown_mt5(self)->None:
        self.connected=False
    def get_symbol_info(self,symbol:str):
        return None
    def get_account_info(self):
        return {}
    def calculate_lot_size(self,*args,**kwargs):
        raise RuntimeError("MT5 execution adapter is disabled")
    def execute_trade(self,*args,**kwargs):
        return {"success":False,"error":"Live broker execution is disabled","orders_submitted":0}

MT5BrokerService=MT5TradingService
mt5_trading_service=MT5TradingService()
