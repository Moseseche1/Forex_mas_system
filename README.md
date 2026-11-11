# Olympus FX — Forex Multi-Agent Research Platform

A safe foundation for a multi-agent forex research and paper-trading platform.

## Current safety boundary

- **Paper/research mode only.** This API does not submit broker orders.
- Live execution is deliberately disabled and must not be enabled by changing a UI setting.
- No broker credentials are required to run the API.
- No synthetic market prices are presented as live prices.
- The API does not auto-start a trading loop on application startup.

The previous implementation mixed generated/demo prices with live MT5 execution, generated random P&L statistics, and attempted broker initialization during startup. Those behaviours are unsuitable for real-money use and have been removed from the API entry point.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set KILL_SWITCH_TOKEN to a long random value in .env
uvicorn main:app --reload
```

Open http://localhost:8000/dashboard or http://localhost:8000/docs.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

## Tests

```bash
pytest -q
```

## API

- `GET /health` — process health
- `GET /status` — runtime and agent status
- `GET /agent/{id}` — agent metadata
- `POST /start` — start research/paper runtime state only
- `POST /stop` — stop runtime
- `POST /kill` — emergency stop; requires `X-Admin-Token`

## Next engineering gates

1. Add deterministic historical-data adapters with provenance and freshness checks.
2. Build reproducible backtests with spread, commission, slippage, and out-of-sample validation.
3. Implement portfolio-level risk controls and an immutable audit trail.
4. Run MT5 connectivity from a dedicated Windows host/worker; the Linux API container must not pretend to be an MT5 terminal.
5. Paper-trade and independently validate before considering any live execution integration.

No profitability or accuracy is guaranteed. Never put API keys, broker passwords, or account identifiers in source control or chat.
