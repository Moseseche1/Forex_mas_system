# Olympus FX — Research and Paper-Trading Platform

## Safety status
This release is **research/paper-only**. It does not submit broker orders. Legacy MT5 execution modules are disabled compatibility shims. Do not add broker credentials to GitHub or the API container.

## Capabilities
- FastAPI health, status and dashboard endpoints.
- Seven role-labelled research agents (runtime registry only; no autonomous LLM agents yet).
- Deterministic EMA crossover and RSI calculations.
- Position sizing from broker tick economics, rounding volume down and rejecting a minimum lot that breaches the risk budget.
- OHLC backtesting with next-bar entries, spread, slippage, commission and conservative stop-first handling when both stop and target are touched.
- CSV historical-price ingestion with OHLC validation.
- Optional OANDA read-only quotes and completed candles. Configure `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, and optionally `OANDA_API_URL`. Practice API is the default. No order endpoints are implemented.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```
Open `http://localhost:8000/dashboard` or `/docs`.

## Docker
```bash
cp .env.example .env
docker compose up --build
```

## Tests
```bash
pytest -q
```

## API
- `GET /health`, `GET /status`, `GET /agent/{id}`
- `POST /start`, `POST /stop`, `POST /kill` (kill requires `X-Admin-Token`)
- `POST /risk/position-size` — risk-limited lot sizing
- `POST /backtest` — deterministic OHLC simulation, no broker interaction
- `POST /signals/ema-crossover` — deterministic research signals
- `POST /agents/analyze` — explainable specialist-agent consensus and risk veto
- `GET /market/status`, `/market/quote/{symbol}`, `/market/candles/{symbol}` — optional OANDA read-only data; returns 503 when unconfigured

## Multi-agent research
The initial ensemble is deterministic and explainable: trend (EMA), momentum (RSI), volatility and risk specialists feed a portfolio consensus. The executive role reports the result. It is not yet an autonomous LLM multi-agent system; no agent can submit orders.

## Backtesting caveats
The backtester is a research baseline, not evidence of a profitable strategy. Supply broker-correct tick size/value, lot constraints, account-currency conversion, spread, commission and slippage. It does not yet model swaps, financing, partial fills, variable spreads, latency, margin liquidation, multi-currency conversion or portfolio-level correlated exposure. Results must be independently reconciled with broker data before use.

## Remaining production gates
1. Validate OANDA market-data freshness, rate limits and provider outages against live read-only practice credentials.
2. Add robust account-currency conversion and broker contract metadata for each symbol.
3. Add a true paper execution ledger and event/audit store.
4. Add portfolio exposure/correlation and daily loss circuit breakers.
5. Implement the seven agents as deterministic, testable modules with an orchestrator; current role list is metadata, not autonomous agents.
6. Run out-of-sample and walk-forward research with multiple market regimes.
7. Only after independent review, consider a separately isolated MT5 Windows worker. It must have its own hard risk gate, idempotency, reconciliation, kill switch and explicit operator enablement.

No return or accuracy is promised. Do not run live funds based on backtest output.
