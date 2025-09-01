import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
from config.settings import settings

logger = logging.getLogger(__name__)

class MT5BrokerService:
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.initialized = False
        
    def initialize_mt5(self) -> bool:
        """Initialize MetaTrader5 connection"""
        try:
            if not self.initialized:
                if mt5.initialize():
                    self.initialized = True
                    logger.info("MT5 initialized successfully")
                    
                    # Login to account
                    account_login = int(os.getenv('MT5_ACCOUNT_LOGIN', 0))
                    account_password = os.getenv('MT5_ACCOUNT_PASSWORD', '')
                    account_server = os.getenv('MT5_ACCOUNT_SERVER', '')
                    
                    if account_login and account_password:
                        if mt5.login(account_login, account_password, account_server):
                            self.connected = True
                            self.account_info = mt5.account_info()
                            logger.info(f"Connected to MT5 account: {account_login}")
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
        """Shutdown MT5 connection"""
        if self.initialized:
            mt5.shutdown()
            self.connected = False
            self.initialized = False
            logger.info("MT5 shutdown complete")
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.connected:
            return None
            
        try:
            account = mt5.account_info()
            return {
                'balance': account.balance,
                'equity': account.equity,
                'profit': account.profit,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'leverage': account.leverage,
                'currency': account.currency,
                'server': account.server
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.connected:
            return None
            
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
                
            tick = mt5.symbol_info_tick(symbol)
            return {
                'symbol': symbol,
                'bid': tick.bid if tick else info.bid,
                'ask': tick.ask if tick else info.ask,
                'spread': info.spread,
                'points': info.point,
                'digits': info.digits,
                'trade_allowed': info.trade_allowed,
                'min_lot': info.volume_min,
                'max_lot': info.volume_max,
                'lot_step': info.volume_step
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol: str, trade_type: str, volume: float, 
                     stop_loss: float = 0.0, take_profit: float = 0.0) -> Dict:
        """Execute a trade"""
        if not self.connected:
            return {'success': False, 'error': 'Not connected to MT5', 'simulated': True}
            
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info or not symbol_info['trade_allowed']:
                return {'success': False, 'error': f'Symbol {symbol} not available for trading'}
            
            # Prepare trade request
            price = symbol_info['ask'] if trade_type.upper() == 'BUY' else symbol_info['bid']
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if trade_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 1000001,
                "comment": "AI Trading System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    'success': True,
                    'order_id': result.order,
                    'price': result.price,
                    'volume': result.volume,
                    'profit': result.profit,
                    'comment': result.comment
                }
            else:
                return {
                    'success': False,
                    'error': f"Trade failed: {result.comment}",
                    'retcode': result.retcode
                }
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self.connected:
            return []
            
        try:
            positions = mt5.positions_get()
            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'current_price': pos.price_current,
                    'profit': pos.profit,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'open_time': pd.to_datetime(pos.time, unit='s')
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, ticket: int) -> Dict:
        """Close a position by ticket number"""
        if not self.connected:
            return {'success': False, 'error': 'Not connected to MT5'}
            
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Position not found'}
                
            position = position[0]
            symbol_info = self.get_symbol_info(position.symbol)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": symbol_info['bid'] if position.type == mt5.ORDER_TYPE_BUY else symbol_info['ask'],
                "deviation": 20,
                "magic": 1000001,
                "comment": "AI Close Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {'success': True, 'closed_price': result.price}
            else:
                return {'success': False, 'error': result.comment}
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}

# Global instance
mt5_broker = MT5BrokerService()
