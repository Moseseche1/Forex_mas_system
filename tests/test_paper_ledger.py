from core.paper_ledger import PaperLedger
import pytest

def test_paper_ledger_persists_and_closes(tmp_path):
    path=str(tmp_path/"paper.sqlite3")
    ledger=PaperLedger(path)
    position=ledger.create_position(
        symbol="EURUSD",side="long",volume_lots=.1,entry_price=1.1,
        stop_price=1.099,target_price=1.102,tick_size=.0001,
        tick_value_per_lot=10,commission_per_lot=7,risk_budget=1,
        estimated_loss=.9
    )
    assert position["status"]=="open"
    reopened=PaperLedger(path)
    assert reopened.get_position(position["id"])["symbol"]=="EURUSD"
    closed=reopened.close_position(position["id"],1.101)
    assert closed["status"]=="closed"
    assert closed["realized_pnl"]==pytest.approx(9.3)
    assert reopened.realized_pnl_since("2000-01-01T00:00:00+00:00")==pytest.approx(9.3)
    with pytest.raises(ValueError):
        reopened.close_position(position["id"],1.102)

def test_paper_ledger_missing_position(tmp_path):
    ledger=PaperLedger(str(tmp_path/"paper.sqlite3"))
    assert ledger.close_position("not-found",1.2) is None

def test_daily_equity_baseline_is_immutable(tmp_path):
    ledger=PaperLedger(str(tmp_path/"paper.sqlite3"))
    assert ledger.get_or_set_day_start_equity("2026-11-11",1000)==1000
    assert ledger.get_or_set_day_start_equity("2026-11-11",5000)==1000
    assert ledger.get_or_set_day_start_equity("2026-11-12",5000)==5000
