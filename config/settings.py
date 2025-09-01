import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database Settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/forex_mas')
    
    # Security Settings
    KILL_SWITCH_TOKEN = os.getenv('KILL_SWITCH_TOKEN', 'MySecretToken123!')
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
    
    # Trading Settings
    MAX_LEVERAGE = float(os.getenv('MAX_LEVERAGE', 10))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))
    MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', 0.05))
    TRADING_HOURS = os.getenv('TRADING_HOURS', '24/7')
    
    # System Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', 30))
    HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL', 60))
    
    # Agent Settings
    TOTAL_AGENTS = int(os.getenv('TOTAL_AGENTS', 20))
    AGENT_STRATEGIES = ['trend_following', 'mean_reversion', 'breakout', 'arbitrage', 'volatility']
    
    # Performance Monitoring
    PERFORMANCE_UPDATE_INTERVAL = int(os.getenv('PERFORMANCE_UPDATE_INTERVAL', 300))
    
    @property
    def database_available(self):
        return all([self.REDIS_HOST, self.MONGO_URI])
    
    @property
    def is_production(self):
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

# Global settings instance
settings = Settings()
