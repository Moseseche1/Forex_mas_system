import aiohttp
import logging
from typing import Dict, Optional
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.session = None
        
    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_real_time_data(self, symbol: str = 'EURUSD') -> Optional[Dict]:
        try:
            # Simulated market data for mobile
            return {
                'symbol': symbol,
                'price': 1.0 + random.uniform(-0.01, 0.01),
                'spread': random.uniform(0.0001, 0.0005),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None

market_data_service = MarketDataService()
