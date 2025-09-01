import aiohttp
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os
from config.settings import settings

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.session = None
        
    async def init_session(self):
        """Initialize aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_real_time_data(self, symbol: str = 'EURUSD') -> Optional[Dict]:
        """Get real-time Forex data from Alpha Vantage"""
        try:
            await self.init_session()
            
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': symbol[:3],
                'to_currency': symbol[3:],
                'apikey': self.api_key
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Realtime Currency Exchange Rate' in data:
                        return self._parse_realtime_data(data['Realtime Currency Exchange Rate'])
                return None
                
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str = 'EURUSD', interval: str = '5min', 
                                output_size: str = 'compact') -> Optional[pd.DataFrame]:
        """Get historical Forex data"""
        try:
            await self.init_session()
            
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': symbol[:3],
                'to_symbol': symbol[3:],
                'interval': interval,
                'output_size': output_size,
                'apikey': self.api_key
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Time Series FX (' + interval + ')' in data:
                        return self._parse_historical_data(data['Time Series FX (' + interval + ')'])
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _parse_realtime_data(self, data: Dict) -> Dict:
        """Parse real-time data response"""
        return {
            'symbol': data['1. From_Currency Code'] + data['3. To_Currency Code'],
            'price': float(data['5. Exchange Rate']),
            'bid': float(data['8. Bid Price']),
            'ask': float(data['9. Ask Price']),
            'spread': float(data['9. Ask Price']) - float(data['8. Bid Price']),
            'timestamp': data['6. Last Refreshed'],
            'volume': float(data['10. Volume']),
        }
    
    def _parse_historical_data(self, time_series: Dict) -> pd.DataFrame:
        """Parse historical data into DataFrame"""
        df_data = []
        for timestamp, values in time_series.items():
            df_data.append({
                'timestamp': pd.to_datetime(timestamp),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': float(values['5. volume']) if '5. volume' in values else 0
            })
        
        return pd.DataFrame(df_data).sort_values('timestamp')
    
    async def get_technical_indicators(self, symbol: str, period: int = 14) -> Dict:
        """Calculate technical indicators"""
        historical_data = await self.get_historical_data(symbol, '15min', 'full')
        if historical_data is None or len(historical_data) < period:
            return {}
        
        closes = historical_data['close'].values
        
        # Calculate basic indicators
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
        
        # Simple RSI calculation
        gains = np.where(np.diff(closes) > 0, np.diff(closes), 0)
        losses = np.where(np.diff(closes) < 0, -np.diff(closes), 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return {
            'sma_20': round(sma_20, 5),
            'sma_50': round(sma_50, 5),
            'rsi': round(rsi, 2),
            'current_price': closes[-1],
            'trend': 'bullish' if sma_20 > sma_50 else 'bearish',
            'volatility': np.std(closes[-20:]) if len(closes) >= 20 else 0
        }

# Global instance
market_data_service = MarketDataService()
