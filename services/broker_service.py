import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
from config.settings import settings

# Import error handling utilities
from utils.error_handler import retry_with_backoff, error_handler

logger = logging.getLogger(__name__)

class MT5BrokerService:
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.initialized = False
        self.symbol_info_cache = {}
        self.last_connection_attempt = 0
        self.connection_timeout = 30  # Seconds between connection attempts
        
    @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=10.0)
    def initialize_mt5(self) -> bool:
        """Initialize MetaTrader5 connection with optimized settings and retry logic"""
        try:
            # Rate limiting - don't attempt connection too frequently
            current_time = time.time()
            if current_time - self.last_connection_attempt < self.connection_timeout:
                logger.warning("Connection attempts too frequent, skipping")
                return self.connected
            
            self.last_connection_attempt = current_time
            
            if not self.initialized:
                # Initialize MT5 with optimized settings
                if not mt5.initialize():
                    error_msg = mt5.last_error()
                    logger.error(f"MT5 initialization failed: {error_msg}")
                    raise ConnectionError(f"MT5 initialization failed: {error_msg}")
                    
                # Set MT5 configuration for better performance
                mt5.set_timeout(5000)  # 5 second timeout
                
                # Login to account
                account_login = int(settings.MT5_ACCOUNT_LOGIN or 0)
                account_password = settings.MT5_ACCOUNT_PASSWORD or ''
                account_server = settings.MT5_ACCOUNT_SERVER or ''
                
                if account_login and account_password:
                    if mt5.login(account_login, account_password, account_server):
                        self.connected = True
                        self.initialized = True
                        self.account_info = mt5.account_info()
                        logger.info(f"Connected to MT5 account: {account_login}")
                        
                        # Preload symbol information for faster trading
                        self._preload_symbol_info()
                        return True
                    else:
                        error_msg = mt5.last_error()
                        logger.error(f"MT5 login failed: {error_msg}")
                        raise ConnectionError(f"MT5 login failed: {error_msg}")
                else:
                    logger.warning("MT5 credentials not set, running in simulation mode")
                    self.initialized = True
                    return True
            return self.connected
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            raise
    
    def _preload_symbol_info(self):
        """Preload symbol information for faster access"""
        try:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'XAUUSD']
            for symbol in symbols:
                info = mt5.symbol_info(symbol)
                if info:
                    self.symbol_info_cache[symbol] = {
                        'point': info.point,
                        'digits': info.digits,
                        'volume_min': info.volume_min,
                        'volume_max': info.volume_max,
                        'volume_step': info.volume_step,
                        'trade_allowed': info.trade_allowed
                    }
            logger.info(f"Preloaded info for {len(self.symbol_info_cache)} symbols")
        except Exception as e:
            logger.error(f"Symbol preloading failed: {e}")
    
    def shutdown_mt5(self):
        """Shutdown MT5 connection"""
        if self.initialized:
            try:
                mt5.shutdown()
                self.connected = False
                self.initialized = False
                self.symbol_info_cache = {}
                logger.info("MT5 shutdown complete")
            except Exception as e:
                logger.error(f"MT5 shutdown error: {e}")
    
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    def get_account_info(self) -> Optional[Dict]:
        """Get account information with retry logic"""
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
                'server': account.server,
                'name': account.name,
                'login': account.login
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information with caching and retry logic"""
        if not self.connected:
            return None
            
        try:
            # Check cache first
            if symbol in self.symbol_info_cache:
                cached_info = self.symbol_info_cache[symbol]
                # Get current tick for real-time prices
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return {
                        **cached_info,
                        'symbol': symbol,
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'spread': tick.ask - tick.bid,
                        'last_update': datetime.utcnow().isoformat()
                    }
            
            # Fallback to direct query
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            
            if info and tick:
                symbol_info = {
                    'symbol': symbol,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'spread': tick.ask - tick.bid,
                    'point': info.point,
                    'digits': info.digits,
                    'trade_allowed': info.trade_allowed,
                    'min_lot': info.volume_min,
                    'max_lot': info.volume_max,
                    'lot_step': info.volume_step,
                    'last_update': datetime.utcnow().isoformat()
                }
                
                # Update cache
                self.symbol_info_cache[symbol] = {
                    'point': info.point,
                    'digits': info.digits,
                    'volume_min': info.volume_min,
                    'volume_max': info.volume_max,
                    'volume_step': info.volume_step,
                    'trade_allowed': info.trade_allowed
                }
                
                return symbol_info
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=10.0)
    def execute_trade(self, symbol: str, trade_type: str, volume: float, 
                     stop_loss: float = 0.0, take_profit: float = 0.0,
                     magic: int = 1000001, comment: str = "AI Trading System") -> Dict:
        """Execute a trade with optimized error handling and retry logic"""
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
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_result = {
                    'success': True,
                    'order_id': result.order,
                    'deal_id': result.deal,
                    'price': result.price,
                    'volume': result.volume,
                    'profit': result.profit,
                    'commission': result.commission,
                    'swap': result.swap,
                    'comment': result.comment,
                    'retcode': result.retcode,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                logger.info(f"Trade executed successfully: {trade_result}")
                return trade_result
            else:
                error_msg = self._get_error_message(result.retcode)
                trade_result = {
                    'success': False,
                    'error': error_msg,
                    'retcode': result.retcode,
                    'comment': result.comment,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                logger.warning(f"Trade failed: {trade_result}")
                return trade_result
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'success': False, 'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def _get_error_message(self, retcode: int) -> str:
        """Convert MT5 error code to human-readable message"""
        error_messages = {
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Only part of the request was completed",
            10011: "Request processing error",
            10012: "Request canceled by timeout",
            10013: "Invalid request",
            10014: "Invalid volume",
            10015: "Invalid price",
            10016: "Invalid stops",
            10017: "Trade disabled",
            10018: "Market closed",
            10019: "Not enough money",
            10020: "Price changed",
            10021: "Too many requests",
            10022: "No changes",
            10023: "Server busy",
            10024: "Client busy",
            10025: "Order locked",
            10026: "Frozen account",
            10027: "Invalid account",
            10028: "Trade timeout",
            10029: "Invalid trade volume",
            10030: "Invalid trade price",
            10031: "Invalid trade stops",
            10032: "Trade disabled",
            10033: "Market closed",
            10034: "Not enough money",
            10035: "Price changed",
            10036: "Too many requests",
            10037: "No changes",
            10038: "Server busy",
            10039: "Client busy",
            10040: "Order locked",
            10041: "Frozen account",
            10042: "Invalid account",
            10043: "Trade timeout"
        }
        return error_messages.get(retcode, f"Unknown error: {retcode}")
    
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions with retry logic"""
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
                    'magic': pos.magic,
                    'comment': pos.comment,
                    'open_time': pd.to_datetime(pos.time, unit='s')
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def close_position(self, ticket: int) -> Dict:
        """Close a position by ticket number with retry logic"""
        if not self.connected:
            return {'success': False, 'error': 'Not connected to MT5'}
            
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Position not found'}
                
            position = position[0]
            symbol_info = self.get_symbol_info(position.symbol)
            
            if not symbol_info:
                return {'success': False, 'error': 'Symbol info not available'}
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": symbol_info['bid'] if position.type == mt5.ORDER_TYPE_BUY else symbol_info['ask'],
                "deviation": 20,
                "magic": position.magic,
                "comment": f"Closed by AI: {position.comment}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {'success': True, 'closed_price': result.price, 'profit': result.profit}
            else:
                return {'success': False, 'error': self._get_error_message(result.retcode)}
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_service_status(self) -> Dict[str, any]:
        """Get the current status of the broker service"""
        return {
            'connected': self.connected,
            'initialized': self.initialized,
            'cached_symbols_count': len(self.symbol_info_cache),
            'last_connection_attempt': self.last_connection_attempt,
            'account_login': settings.MT5_ACCOUNT_LOGIN if self.connected else None,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global instance
mt5_broker = MT5BrokerService()
