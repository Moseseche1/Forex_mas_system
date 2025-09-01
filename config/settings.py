import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    KILL_SWITCH_TOKEN = os.getenv('KILL_SWITCH_TOKEN', 'default')

settings = Settings()
