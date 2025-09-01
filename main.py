
```python
from fastapi import FastAPI, HTTPException
import redis
import pymongo
import time
import numpy as np
import random
import logging
import asyncio
from datetime import datetime
import os
import json

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
KILL_SWITCH_TOKEN = os.getenv('KILL_SWITCH_TOKEN', 'default_token')

# Initialize
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client['forex_mas']

class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.active = True
        self.emotion_state = {'confidence': 0.5, 'caution': 0.3}
        
    def update_emotions(self, market_data: float):
        self.emotion_state['confidence'] = min(1.0, market_data * 0.8)
        redis_client.set(f'agent_{self.agent_id}', json.dumps(self.emotion_state))

class TradingSystem:
    def __init__(self):
        self.agents = [Agent(i) for i in range(20)]
    
    async def run_cycle(self):
        for agent in self.agents:
            if agent.active:
                market_data = random.random()
                agent.update_emotions(market_data)
        logger.info("Trading cycle completed")

trading_system = TradingSystem()

app = FastAPI()

@app.get('/status')
async def status():
    return {
        'status': 'running',
        'agents_active': sum(1 for a in trading_system.agents if a.active),
        'timestamp': datetime.utcnow().isoformat()
    }

@app.post('/kill')
async def kill_switch(auth_token: str):
    if auth_token != KILL_SWITCH_TOKEN:
        raise HTTPException(status_code=401, detail='Invalid token')
    
    for agent in trading_system.agents:
        agent.active = False
    
    return {'status': 'shutdown'}

async def trading_loop():
    while True:
        await trading_system.run_cycle()
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup():
    asyncio.create_task(trading_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
