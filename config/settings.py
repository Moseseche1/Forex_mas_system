"""Validated application settings. Live execution is disabled by default."""
import os
from dotenv import load_dotenv

load_dotenv()

def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

class Settings:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/forex_mas")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
    UPDATE_INTERVAL = max(5, int(os.getenv("UPDATE_INTERVAL", "30")))
    TOTAL_AGENTS = max(1, min(50, int(os.getenv("TOTAL_AGENTS", "7"))))
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.03"))
    MT5_ACCOUNT_LOGIN = os.getenv("MT5_ACCOUNT_LOGIN", "")
    MT5_ACCOUNT_PASSWORD = os.getenv("MT5_ACCOUNT_PASSWORD", "")
    MT5_ACCOUNT_SERVER = os.getenv("MT5_ACCOUNT_SERVER", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
    ALLOW_LIVE_TRADING = _bool("ALLOW_LIVE_TRADING", False)
    KILL_SWITCH_TOKEN = os.getenv("KILL_SWITCH_TOKEN", "")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
    OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
    OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
    FX_SYMBOLS = os.getenv("FX_SYMBOLS", "EURUSD,GBPUSD,USDJPY").split(",")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

settings = Settings()
