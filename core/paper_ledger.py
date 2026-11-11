"""Durable SQLite ledger for simulated positions only; never connects to a broker."""
from __future__ import annotations
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

class PaperLedger:
    def __init__(self, database_path: str):
        self.database_path = database_path
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('long','short')),
                    volume_lots REAL NOT NULL CHECK(volume_lots > 0),
                    entry_price REAL NOT NULL CHECK(entry_price > 0),
                    stop_price REAL NOT NULL CHECK(stop_price > 0),
                    target_price REAL,
                    tick_size REAL NOT NULL CHECK(tick_size > 0),
                    tick_value_per_lot REAL NOT NULL CHECK(tick_value_per_lot > 0),
                    commission_per_lot REAL NOT NULL DEFAULT 0,
                    risk_budget REAL NOT NULL,
                    estimated_loss REAL NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('open','closed')),
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    exit_price REAL,
                    realized_pnl REAL
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_paper_status ON paper_positions(status)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_paper_opened ON paper_positions(opened_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS paper_daily_state (
                utc_day TEXT PRIMARY KEY,
                day_start_equity REAL NOT NULL CHECK(day_start_equity > 0)
            )""")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.database_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _serialize(row):
        return dict(row) if row else None

    def create_position(self, *, symbol, side, volume_lots, entry_price, stop_price,
                        target_price, tick_size, tick_value_per_lot, commission_per_lot,
                        risk_budget, estimated_loss):
        position_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as db:
            db.execute("""
                INSERT INTO paper_positions
                (id,symbol,side,volume_lots,entry_price,stop_price,target_price,
                 tick_size,tick_value_per_lot,commission_per_lot,risk_budget,
                 estimated_loss,status,opened_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (position_id,symbol,side,volume_lots,entry_price,stop_price,target_price,
                  tick_size,tick_value_per_lot,commission_per_lot,risk_budget,
                  estimated_loss,"open",now))
            row = db.execute("SELECT * FROM paper_positions WHERE id=?", (position_id,)).fetchone()
        return self._serialize(row)

    def get_position(self, position_id):
        with self._db() as db:
            return self._serialize(db.execute(
                "SELECT * FROM paper_positions WHERE id=?", (position_id,)
            ).fetchone())

    def list_positions(self, status=None):
        with self._db() as db:
            if status:
                rows=db.execute("SELECT * FROM paper_positions WHERE status=? ORDER BY opened_at DESC",(status,)).fetchall()
            else:
                rows=db.execute("SELECT * FROM paper_positions ORDER BY opened_at DESC").fetchall()
        return [self._serialize(row) for row in rows]


    def get_or_set_day_start_equity(self, utc_day, proposed_equity):
        """Persist the first declared daily equity; later requests cannot raise the loss limit."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT day_start_equity FROM paper_daily_state WHERE utc_day=?",(utc_day,)).fetchone()
            if row:
                return float(row["day_start_equity"])
            db.execute("INSERT INTO paper_daily_state(utc_day,day_start_equity) VALUES(?,?)",
                       (utc_day,float(proposed_equity)))
            return float(proposed_equity)

    def realized_pnl_since(self, iso_timestamp):
        with self._db() as db:
            row=db.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) AS pnl FROM paper_positions WHERE status='closed' AND closed_at>=?",
                (iso_timestamp,)
            ).fetchone()
        return float(row["pnl"])

    def close_position(self, position_id, exit_price):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT * FROM paper_positions WHERE id=?",(position_id,)).fetchone()
            if not row:
                db.rollback()
                return None
            if row["status"] != "open":
                db.rollback()
                raise ValueError("Paper position is already closed")
            direction=1 if row["side"]=="long" else -1
            gross=(exit_price-row["entry_price"])/row["tick_size"]*row["tick_value_per_lot"]*row["volume_lots"]*direction
            commission=row["commission_per_lot"]*row["volume_lots"]
            pnl=gross-commission
            now=datetime.now(timezone.utc).isoformat()
            db.execute(
                "UPDATE paper_positions SET status='closed',closed_at=?,exit_price=?,realized_pnl=? WHERE id=? AND status='open'",
                (now,exit_price,pnl,position_id)
            )
            updated=db.execute("SELECT * FROM paper_positions WHERE id=?",(position_id,)).fetchone()
            db.commit()
        return self._serialize(updated)
