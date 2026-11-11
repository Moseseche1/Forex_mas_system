"""Read-only OANDA market-data adapter. Fails closed when credentials are absent."""
from datetime import datetime, timezone
from math import isfinite
import logging
import httpx
from config.settings import settings

logger=logging.getLogger(__name__)

class MarketDataUnavailable(RuntimeError):
    pass

class MarketDataService:
    def __init__(self):
        self.base_url=getattr(settings,"OANDA_API_URL","https://api-fxpractice.oanda.com").rstrip("/")
        self.timeout=httpx.Timeout(10.0,connect=5.0)

    def _credentials(self):
        token=settings.OANDA_API_KEY
        account=settings.OANDA_ACCOUNT_ID
        if not token or not account:
            raise MarketDataUnavailable("OANDA read-only market data is not configured")
        return token,account

    @staticmethod
    def _instrument(symbol:str)->str:
        clean=symbol.strip().upper().replace("_","")
        if len(clean)!=6 or not clean.isalpha():
            raise ValueError("FX symbol must be a six-letter currency pair, e.g. EURUSD")
        return clean[:3]+"_"+clean[3:]

    async def get_real_time_data(self,symbol:str="EURUSD")->dict:
        token,account=self._credentials()
        instrument=self._instrument(symbol)
        headers={"Authorization":f"Bearer {token}","Accept":"application/json"}
        url=f"{self.base_url}/v3/accounts/{account}/pricing"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response=await client.get(url,headers=headers,params={"instruments":instrument})
            response.raise_for_status()
            payload=response.json()
        prices=payload.get("prices") or []
        if not prices:
            raise MarketDataUnavailable(f"No quote returned for {instrument}")
        quote=prices[0]
        bids=quote.get("bids") or []; asks=quote.get("asks") or []
        if not bids or not asks:
            raise MarketDataUnavailable(f"Incomplete bid/ask quote for {instrument}")
        bid=float(bids[0]["price"]); ask=float(asks[0]["price"])
        if not isfinite(bid) or not isfinite(ask) or bid<=0 or ask<bid:
            raise MarketDataUnavailable("Provider returned invalid bid/ask values")
        retrieved=datetime.now(timezone.utc)
        source_time=quote.get("time")
        fresh=False
        if source_time:
            try:
                parsed=datetime.fromisoformat(source_time.replace("Z","+00:00"))
                if parsed.tzinfo is None: parsed=parsed.replace(tzinfo=timezone.utc)
                age=(retrieved-parsed.astimezone(timezone.utc)).total_seconds()
                fresh=-5<=age<=60
            except ValueError:
                fresh=False
        return {"symbol":symbol.upper(),"bid":bid,"ask":ask,"spread":ask-bid,
          "mid":(bid+ask)/2,"timestamp":source_time,"is_fresh":fresh,
          "source":"oanda","tradeable":bool(quote.get("tradeable",False)),
          "retrieved_at":retrieved.isoformat(),"read_only":True}

    async def get_historical_data(self,symbol:str="EURUSD",granularity:str="M5",count:int=500)->list[dict]:
        token,account=self._credentials()
        instrument=self._instrument(symbol)
        if granularity not in {"M1","M5","M15","M30","H1","H4","D"}:
            raise ValueError("Unsupported candle granularity")
        if not 10<=count<=5000:
            raise ValueError("count must be between 10 and 5000")
        headers={"Authorization":f"Bearer {token}","Accept":"application/json"}
        url=f"{self.base_url}/v3/instruments/{instrument}/candles"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response=await client.get(url,headers=headers,params={"granularity":granularity,"count":count,"price":"M"})
            response.raise_for_status()
            payload=response.json()
        result=[]
        for candle in payload.get("candles",[]):
            if not candle.get("complete"): continue
            mid=candle.get("mid") or {}
            if not all(k in mid for k in ("o","h","l","c")): continue
            o,h,l,c=(float(mid[k]) for k in ("o","h","l","c"))
            if not all(isfinite(x) and x>0 for x in (o,h,l,c)) or h<max(o,c) or l>min(o,c) or h<l:
                continue
            result.append({"timestamp":candle["time"],"open":o,"high":h,"low":l,"close":c,
              "volume":int(candle.get("volume",0)),"source":"oanda"})
        if not result:
            raise MarketDataUnavailable("Provider returned no complete historical candles")
        return result

    async def close_session(self):
        return None

    def get_service_status(self):
        configured=bool(settings.OANDA_API_KEY and settings.OANDA_ACCOUNT_ID)
        return {"provider":"oanda","configured":configured,"mode":"read_only",
          "synthetic_fallback":False,"timestamp":datetime.now(timezone.utc).isoformat()}

market_data_service=MarketDataService()
