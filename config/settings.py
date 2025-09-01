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
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default_jwt_secret')
    
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
    
    # API Keys
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
    OANDA_API_KEY = os.getenv('OANDA_API_KEY', '')
    OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', '')
    FX_SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP']
    DATA_UPDATE_INTERVAL = int(os.getenv('DATA_UPDATE_INTERVAL', 60))
    
    # MT5 Settings
    MT5_ACCOUNT_LOGIN = os.getenv('MT5_ACCOUNT_LOGIN', '')
    MT5_ACCOUNT_PASSWORD = os.getenv('MT5_ACCOUNT_PASSWORD', '')
    MT5_ACCOUNT_SERVER = os.getenv('MT5_ACCOUNT_SERVER', '')
    
    # Polymorphic Evolution Settings
    EVOLUTION_CYCLE_INTERVAL = int(os.getenv('EVOLUTION_CYCLE_INTERVAL', 100))
    GENETIC_POPULATION_SIZE = int(os.getenv('GENETIC_POPULATION_SIZE', 50))
    MUTATION_RATE = float(os.getenv('MUTATION_RATE', 0.15))
    CROSSOVER_RATE = float(os.getenv('CROSSOVER_RATE', 0.7))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.01))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.3))
    MAX_CAUTION_THRESHOLD = float(os.getenv('MAX_CAUTION_THRESHOLD', 0.7))
    
    # Reinforcement Learning Settings
    DRL_HIDDEN_SIZE = int(os.getenv('DRL_HIDDEN_SIZE', 64))
    DRL_MEMORY_SIZE = int(os.getenv('DRL_MEMORY_SIZE', 10000))
    DRL_BATCH_SIZE = int(os.getenv('DRL_BATCH_SIZE', 32))
    DRL_GAMMA = float(os.getenv('DRL_GAMMA', 0.95))
    DRL_EPSILON_DECAY = float(os.getenv('DRL_EPSILON_DECAY', 0.995))
    
    @property
    def database_available(self):
        return all([self.REDIS_HOST, self.MONGO_URI])
    
    @property
    def is_production(self):
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

# Global settings instance
settings = Settings()
