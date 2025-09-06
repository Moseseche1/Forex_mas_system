import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from config import settings

# Import error handling utilities
from utils.error_handler import retry_with_backoff, error_handler

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_timeout = 60  # seconds
        self.last_update = {}
        self.historical_data = {}
        self.technical_indicators = {}
        
    @retry_with_backoff(max_retries=3, initial_delay=1.0, max_delay=5.0)
    async def init_session(self):
        """Initialize aiohttp session with retry logic"""
        if self.session is None or self.session.closed:
            try:
                timeout = aiohttp.ClientTimeout(total=10.0, connect=5.0)
                self.session = aiohttp.ClientSession(timeout=timeout)
                logger.info("Market data session initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize market data session: {e}")
                raise ConnectionError(f"Market data session initialization failed: {e}")
            
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    async def close_session(self):
        """Close aiohttp session with retry logic"""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                self.session = None
                logger.info("Market data session closed successfully")
            except Exception as e:
                logger.error(f"Error closing market data session: {e}")
                raise
    
    def _get_cache_key(self, symbol: str, function: str, timeframe: str = None) -> str:
        """Generate cache key for market data"""
        if timeframe:
            return f"{symbol}_{function}_{timeframe}"
        return f"{symbol}_{function}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key in self.cache and cache_key in self.last_update:
            return (datetime.utcnow() - self.last_update[cache_key]).total_seconds() < self.cache_timeout
        return False
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0, max_delay=10.0)
    async def get_real_time_data(self, symbol: str = 'EURUSD') -> Optional[Dict]:
        """Get real-time Forex data with retry logic and caching"""
        cache_key = self._get_cache_key(symbol, "realtime")
        
        # Return cached data if valid
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached data for {symbol}")
            return self.cache[cache_key]
        
        try:
            await self.init_session()
            
            # For demo purposes - replace with actual API call
            # Example: Alpha Vantage, OANDA, or other market data provider
            demo_data = await self._get_demo_market_data(symbol)
            
            # Cache the results
            self.cache[cache_key] = demo_data
            self.last_update[cache_key] = datetime.utcnow()
            
            logger.info(f"Real-time data retrieved for {symbol}")
            return demo_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            # Return fallback data instead of failing completely
            return self._get_fallback_data(symbol)
    
    @retry_with_backoff(max_retries=2, initial_delay=2.0)
    async def get_historical_data(self, symbol: str = 'EURUSD', interval: str = '5min', 
                                output_size: str = 'compact') -> Optional[pd.DataFrame]:
        """Get historical Forex data with retry logic"""
        cache_key = self._get_cache_key(symbol, "historical", interval)
        
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached historical data for {symbol}")
            return self.cache[cache_key]
        
        try:
            await self.init_session()
            
            # For demo purposes - replace with actual historical data API
            historical_data = await self._get_demo_historical_data(symbol, interval, output_size)
            
            # Cache the results
            self.cache[cache_key] = historical_data
            self.last_update[cache_key] = datetime.utcnow()
            
            logger.info(f"Historical data retrieved for {symbol} ({interval})")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return self._get_fallback_historical_data(symbol)
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    async def get_technical_indicators(self, symbol: str = 'EURUSD', period: int = 14) -> Dict[str, float]:
        """Calculate technical indicators with retry logic"""
        cache_key = self._get_cache_key(symbol, "technical", str(period))
        
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached technical indicators for {symbol}")
            return self.cache[cache_key]
        
        try:
            # Get historical data for calculations
            historical_data = await self.get_historical_data(symbol, '15min', 'full')
            
            if historical_data is None or len(historical_data) < period:
                logger.warning(f"Insufficient data for technical indicators for {symbol}")
                return self._get_fallback_technical_indicators()
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(historical_data, period)
            
            # Cache the results
            self.cache[cache_key] = indicators
            self.last_update[cache_key] = datetime.utcnow()
            
            logger.info(f"Technical indicators calculated for {symbol}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return self._get_fallback_technical_indicators()
    
    async def _get_demo_market_data(self, symbol: str) -> Dict:
        """Generate demo market data (replace with real API call)"""
        # Simulate some market movement
        base_price = 1.0 + random.uniform(-0.1, 0.1)
        spread = random.uniform(0.0001, 0.0005)
        
        return {
            'symbol': symbol,
            'bid': base_price - spread/2,
            'ask': base_price + spread/2,
            'spread': spread,
            'price': base_price,
            'volume': random.uniform(0.5, 1.5),
            'timestamp': datetime.utcnow().isoformat(),
            'volatility': random.uniform(0.05, 0.15),
            'trend_strength': random.uniform(-0.8, 0.8),
            'liquidity': random.uniform(0.7, 1.0),
            'source': 'demo'
        }
    
    async def _get_demo_historical_data(self, symbol: str, interval: str, output_size: str) -> pd.DataFrame:
        """Generate demo historical data (replace with real API call)"""
        # Generate synthetic price data
        periods = 100 if output_size == 'compact' else 1000
        base_price = 1.0
        volatility = 0.001
        
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq=interval)
        prices = []
        
        current_price = base_price
        for _ in range(periods):
            change = random.gauss(0, volatility)
            current_price *= (1 + change)
            prices.append(current_price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 - random.uniform(0, 0.0005)) for p in prices],
            'high': [p * (1 + random.uniform(0, 0.001)) for p in prices],
            'low': [p * (1 - random.uniform(0, 0.001)) for p in prices],
            'close': prices,
            'volume': [random.uniform(0.5, 1.5) for _ in range(periods)]
        })
        
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        closes = df['close'].values
        
        # Simple Moving Averages
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
        
        # RSI Calculation
        if len(closes) >= period + 1:
            changes = np.diff(closes)
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50
        
        # MACD (simplified)
        if len(closes) >= 26:
            ema_12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
            ema_26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
        else:
            macd = 0
        
        # Volatility
        volatility = np.std(closes[-20:]) if len(closes) >= 20 else 0.001
        
        return {
            'sma_20': float(sma_20),
            'sma_50': float(sma_50),
            'rsi': float(rsi),
            'macd': float(macd),
            'volatility': float(volatility),
            'trend': 'bullish' if sma_20 > sma_50 else 'bearish',
            'trend_strength': float(abs(sma_20 - sma_50) / sma_50 * 100),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_fallback_data(self, symbol: str) -> Dict:
        """Provide fallback data when primary source fails"""
        return {
            'symbol': symbol,
            'bid': 1.0,
            'ask': 1.0002,
            'spread': 0.0002,
            'price': 1.0001,
            'volume': 1.0,
            'timestamp': datetime.utcnow().isoformat(),
            'volatility': 0.1,
            'trend_strength': 0.0,
            'liquidity': 0.8,
            'source': 'fallback',
            'warning': 'Using fallback data - primary source unavailable'
        }
    
    def _get_fallback_historical_data(self, symbol: str) -> pd.DataFrame:
        """Provide fallback historical data"""
        dates = pd.date_range(end=datetime.utcnow(), periods=50, freq='5min')
        prices = [1.0 + random.uniform(-0.01, 0.01) for _ in range(50)]
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.0005 for p in prices],
            'low': [p * 0.9995 for p in prices],
            'close': prices,
            'volume': [1.0 for _ in range(50)]
        })
    
    def _get_fallback_technical_indicators(self) -> Dict[str, float]:
        """Provide fallback technical indicators"""
        return {
            'sma_20': 1.0,
            'sma_50': 1.0,
            'rsi': 50.0,
            'macd': 0.0,
            'volatility': 0.1,
            'trend': 'neutral',
            'trend_strength': 0.0,
            'timestamp': datetime.utcnow().isoformat(),
            'warning': 'Using fallback technical indicators'
        }
    
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    async def get_multiple_symbols_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get data for multiple symbols with retry logic"""
        results = {}
        
        for symbol in symbols:
            try:
                data = await self.get_real_time_data(symbol)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                results[symbol] = self._get_fallback_data(symbol)
        
        return results
    
    def clear_cache(self, symbol: str = None):
        """Clear cached market data"""
        if symbol:
            # Clear all cache entries for this symbol
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(symbol)]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.last_update.pop(key, None)
            logger.info(f"Cache cleared for symbol: {symbol}")
        else:
            # Clear all cache
            self.cache.clear()
            self.last_update.clear()
            logger.info("All market data cache cleared")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the market data service"""
        return {
            'session_active': self.session is not None and not self.session.closed,
            'cached_items': len(self.cache),
            'cache_timeout': self.cache_timeout,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global instance
market_data_service = MarketDataService()
