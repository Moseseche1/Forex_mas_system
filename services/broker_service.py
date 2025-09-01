import MetaTrader5 as mt5
import logging
from typing import Dict, List, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class MT5BrokerService:
    def __init__(self):
        self.connected = False
        self.initialized = False
        
    def initialize_mt5(self) -> bool:
        try:
            if not self.initialized:
                if mt5.initialize():
                    self.initialized = True
                    
                    # Login to account
                    if settings.MT5_ACCOUNT_LOGIN and settings.MT5_ACCOUNT_PASSWORD:
                        if mt5.login(settings.MT5_ACCOUNT_LOGIN, settings.MT5_ACCOUNT_PASSWORD, settings.MT5_ACCOUNT_SERVER):
                            self.connected = True
                            logger.info(f"Connected to MT5 account: {settings.MT5_ACCOUNT_LOGIN}")
                            return True
                        else:
                            logger.error("MT5 login failed")
                            return False
                    else:
                        logger.warning("MT5 credentials not set, running in simulation mode")
                        return True
                else:
                    logger.error("MT5 initialization failed")
                    return False
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def shutdown_mt5(self):
        if self.initialized:
            mt5.shutdown()
            self.connected = False
            self.initialized = False
            logger.info("MT5 shutdown complete")
    
    def execute_trade(self, symbol: str, trade_type: str, volume: float) -> Dict:
        if not self.connected:
            return {'success': False, 'error': 'Not connected to MT5', 'simulated': True}
            
        try:
            # Simplified trade execution for mobile
            return {
                'success': True,
                'symbol': symbol,
                'type': trade_type,
                'volume': volume,
                'simulated': not self.connected
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

mt5_broker = MT5BrokerService()
