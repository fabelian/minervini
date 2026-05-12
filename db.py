"""Railway PostgreSQL 캐싱.

같은 (market, as_of, max_stocks, config) 입력의 파이프라인 실행을 DB에서
조회만 하도록 캐싱한다. `DATABASE_URL` 미설정/연결 실패 시 모든 호출이
no-op이 되도록 graceful하게 동작 — 로컬 개발/Railway 미설정 환경에서도
앱은 그대로 굴러간다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

log = logging.getLogger("minervini.db")

try:
    import psycopg
    from psycopg.types.json import Json
    from psycopg_pool import ConnectionPool
    _HAS_PSYCOPG = True
except ImportError:
    _HAS_PSYCOPG = False

_pool: "ConnectionPool | None" = None
_disabled_reason: str = ""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    market          VARCHAR(16)  NOT NULL,
    as_of           DATE         NOT NULL,
    max_stocks      INTEGER      NOT NULL,
    config_hash     CHAR(16)     NOT NULL,
    config_json     JSONB        NOT NULL,
    summary         JSONB        NOT NULL,
    dashboard_png   BYTEA,
    n_stocks        INTEGER,
    client_ip       VARCHAR(64),
    started_at      TIMESTAMPTZ  NOT NULL,
    finished_at     TIMESTAMPTZ  NOT NULL,
    UNIQUE (market, as_of, max_stocks, config_hash)
);
CREATE INDEX IF NOT EXISTS idx_runs_market_asof
    ON pipeline_runs (market, as_of DESC);

-- 기존 배포(컬럼 추가 전)에 대한 자동 마이그레이션
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS client_ip VARCHAR(64);
"""


def init() -> bool:
    """앱 시작 시 1회 호출. 성공 시 True, 비활성/실패 시 False (앱은 정상 동작)."""
    global _pool, _disabled_reason
    if _pool is not None:
        return True
    if not _HAS_PSYCOPG:
        _disabled_reason = "psycopg not installed"
        log.info("db disabled: %s", _disabled_reason)
        return False
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        _disabled_reason = "DATABASE_URL not set"
        log.info("db disabled: %s", _disabled_reason)
        return False
    # 일부 환경(Heroku 잔재 등)이 postgres:// 사용 — 정규화
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    try:
        _pool = ConnectionPool(url, min_size=1, max_size=4, open=True)
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            conn.commit()
        log.info("db enabled")
        return True
    except Exception as e:  # noqa: BLE001
        _disabled_reason = f"{type(e).__name__}: {e}"
        log.warning("db init failed: %s", _disabled_reason)
        _pool = None
        return False


def is_enabled() -> bool:
    return _pool is not None


def status() -> dict[str, Any]:
    return {"enabled": is_enabled(), "reason": _disabled_reason}


def config_hash(cfg) -> str:
    """Config의 분석 파라미터를 16자 hash로. cache_dir(환경마다 다름) 제외."""
    d = asdict(cfg)
    d.pop("cache_dir", None)
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def _norm_date(as_of: str | None) -> date:
    if as_of:
        return date.fromisoformat(as_of[:10])
    return date.today()


def _norm_max_stocks(max_stocks: int | None) -> int:
    return -1 if max_stocks is None else int(max_stocks)


def lookup(market: str, as_of: str | None, max_stocks: int | None,
           cfg_hash: str) -> dict[str, Any] | None:
    if not is_enabled():
        return None
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary, dashboard_png, n_stocks, finished_at
                FROM pipeline_runs
                WHERE market=%s AND as_of=%s AND max_stocks=%s AND config_hash=%s
                """,
                (market, _norm_date(as_of), _norm_max_stocks(max_stocks), cfg_hash),
            )
            row = cur.fetchone()
            if not row:
                return None
            summary, png, n_stocks, finished_at = row
            return {
                "summary": summary,
                "dashboard_png": bytes(png) if png is not None else None,
                "n_stocks": n_stocks,
                "finished_at": finished_at.isoformat() if finished_at else None,
            }
    except Exception as e:  # noqa: BLE001
        log.warning("db lookup failed: %s", e)
        return None


def save(market: str, as_of: str | None, max_stocks: int | None,
         cfg_hash: str, config_json: dict, summary: dict,
         dashboard_png: bytes | None, n_stocks: int,
         started_at: datetime, finished_at: datetime,
         client_ip: str | None = None) -> bool:
    if not is_enabled():
        return False
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_runs
                  (market, as_of, max_stocks, config_hash, config_json,
                   summary, dashboard_png, n_stocks, client_ip,
                   started_at, finished_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (market, as_of, max_stocks, config_hash)
                DO UPDATE SET
                  config_json   = EXCLUDED.config_json,
                  summary       = EXCLUDED.summary,
                  dashboard_png = EXCLUDED.dashboard_png,
                  n_stocks      = EXCLUDED.n_stocks,
                  client_ip     = EXCLUDED.client_ip,
                  started_at    = EXCLUDED.started_at,
                  finished_at   = EXCLUDED.finished_at
                """,
                (market, _norm_date(as_of), _norm_max_stocks(max_stocks),
                 cfg_hash, Json(config_json), Json(summary),
                 dashboard_png, n_stocks,
                 (client_ip or None),
                 started_at, finished_at),
            )
            conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("db save failed: %s", e)
        return False
