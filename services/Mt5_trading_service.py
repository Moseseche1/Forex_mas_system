import MetaTrader5 as mt5
import logging
from typing import Dict, List, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class MT5TradingService:
    def __init__(self):
        self.connected = False
        self.symbols = settings.MT5_SYMBOLS.split(',')
        
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection with optimized settings"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
                
            # Login to account
            if not mt5.login(
                login=settings.MT5_ACCOUNT_LOGIN,
                password=settings.MT5_ACCOUNT_PASSWORD,
                server=settings.MT5_ACCOUNT_SERVER
            ):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
            self.connected = True
            logger.info(f"Connected to MT5 account: {settings.MT5_ACCOUNT_LOGIN}")
            
            # Preload symbol info for faster trading
            for symbol in self.symbols:
                mt5.symbol_select(symbol, True)
                
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get optimized symbol information"""
        try:
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'point': info.point,
                'digits': info.digits,
                'trade_allowed': info.trade_allowed,
                'min_lot': info.volume_min,
                'max_lot': info.volume_max,
                'lot_step': info.volume_step
            }
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def execute_trade(self, symbol: str, trade_type: str, volume: float, 
                     stop_loss: float = 0.0, take_profit: float = 0.0) -> Dict:
        """Execute trade with MT5-optimized parameters"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': 'Symbol not available'}
            
            # Calculate precise lot size
            lot_size = self.calculate_lot_size(volume, symbol_info)
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if trade_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": symbol_info['ask'] if trade_type.upper() == 'BUY' else symbol_info['bid'],
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": settings.MT5_DEVIATION,
                "magic": settings.MT5_MAGIC_NUMBER,
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
                    'error': mt5.last_error(),
                    'retcode': result.retcode
                }
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_lot_size(self, risk_amount: float, symbol_info: Dict) -> float:
        """Calculate precise lot size for MT5"""
        # MT5 has specific lot step requirements
        min_lot = symbol_info['min_lot']
        max_lot = symbol_info['max_lot']
        lot_step = symbol_info['lot_step']
        
        # Calculate raw lot size
        raw_lot = risk_amount
        
        # Apply MT5 constraints
        lot_size = max(min_lot, min(max_lot, raw_lot))
        lot_size = round(lot_size / lot_step) * lot_step  # Round to nearest step
        
        return round(lot_size, 2)
    
    def get_account_info(self) -> Dict:
        """Get comprehensive account info"""
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
            return {}

# Global instance
mt5_trading_service = MT5TradingService()
