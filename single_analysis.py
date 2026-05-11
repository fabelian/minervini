"""단일 종목 돌파매매 분석.

monitor._prepare_one 을 재사용해 한 종목의 SMA / 모멘텀 / 트렌드 템플릿 /
브레이크아웃 / forward return 라벨을 계산하고 트레이딩 의사결정에 쓸 수 있는
형태로 정리해 JSON 직렬화 가능한 dict로 반환한다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import Config
from data import get_ohlcv, get_listing, get_sector_map
from monitor import _prepare_one


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _safe_bool(x) -> bool:
    if x is None or pd.isna(x):
        return False
    return bool(x)


def _series_pts(s: pd.Series) -> list[dict]:
    out: list[dict] = []
    for d, v in s.items():
        if v is None or pd.isna(v):
            continue
        out.append({"date": d.strftime("%Y-%m-%d"), "value": float(v)})
    return out


def detect_market(ticker: str) -> str:
    """KOSPI 6자리 숫자 vs NASDAQ 영문 심볼."""
    t = (ticker or "").strip()
    return "KOSPI" if t.isdigit() and len(t) == 6 else "NASDAQ"


def _trend_template_checks(row: pd.Series) -> list[dict]:
    close = _safe_float(row.get("Close"))
    sma50 = _safe_float(row.get("SMA50"))
    sma150 = _safe_float(row.get("SMA150"))
    sma200 = _safe_float(row.get("SMA200"))
    slope = _safe_float(row.get("sma200_slope"))
    low52 = _safe_float(row.get("low_52w"))
    high52 = _safe_float(row.get("high_52w"))

    checks: list[dict] = []

    def add(label: str, ok: bool, value: str) -> None:
        checks.append({"label": label, "ok": ok, "value": value})

    if close is not None and sma150 is not None:
        add("Close > SMA150", close > sma150, f"{close:.2f} vs {sma150:.2f}")
    else:
        add("Close > SMA150", False, "—")

    if close is not None and sma200 is not None:
        add("Close > SMA200", close > sma200, f"{close:.2f} vs {sma200:.2f}")
    else:
        add("Close > SMA200", False, "—")

    if sma150 is not None and sma200 is not None:
        add("SMA150 > SMA200", sma150 > sma200, f"{sma150:.2f} vs {sma200:.2f}")
    else:
        add("SMA150 > SMA200", False, "—")

    if slope is not None:
        add("SMA200 1개월 상승", slope > 0, f"slope={slope:+.2f}")
    else:
        add("SMA200 1개월 상승", False, "—")

    if sma50 is not None and sma150 is not None and sma200 is not None:
        ok = sma50 > sma150 > sma200
        add("SMA50 > SMA150 > SMA200", ok,
            f"{sma50:.2f} > {sma150:.2f} > {sma200:.2f}")
    else:
        add("SMA50 > SMA150 > SMA200", False, "—")

    if close is not None and sma50 is not None:
        add("Close > SMA50", close > sma50, f"{close:.2f} vs {sma50:.2f}")
    else:
        add("Close > SMA50", False, "—")

    if close is not None and low52 is not None:
        add("저점 대비 +30% 이상", close >= low52 * 1.30,
            f"{close:.2f} vs {low52*1.30:.2f}")
    else:
        add("저점 대비 +30% 이상", False, "—")

    if close is not None and high52 is not None:
        add("신고가 대비 -25% 이내", close >= high52 * 0.75,
            f"{close:.2f} vs {high52*0.75:.2f}")
    else:
        add("신고가 대비 -25% 이내", False, "—")

    return checks


def analyze_ticker(ticker: str, market: str | None = None,
                   as_of: str | None = None,
                   max_breakouts: int = 30,
                   chart_lookback: int = 252) -> dict[str, Any]:
    ticker = (ticker or "").strip()
    if not ticker:
        raise ValueError("ticker가 비어있습니다.")
    market = (market or detect_market(ticker)).upper()
    cfg = Config.for_market(market)

    if as_of:
        end_ts = pd.Timestamp(as_of).normalize()
    else:
        end_ts = pd.Timestamp(datetime.now()).normalize()
    end = end_ts.strftime("%Y-%m-%d")
    start = (end_ts - pd.Timedelta(days=int(cfg.lookback_days * 1.6))).strftime("%Y-%m-%d")

    df = get_ohlcv(ticker, start, end, cfg.cache_dir)
    if df is None or df.empty:
        raise ValueError(f"{market} 종목 {ticker} 의 OHLCV 데이터를 가져올 수 없습니다.")
    df = df.loc[df.index <= end_ts]
    if len(df) < cfg.sma_long + 30:
        raise ValueError(
            f"분석에 필요한 거래일이 부족합니다 (필요 ≥ {cfg.sma_long + 30}, 보유 {len(df)})."
        )

    prepared = _prepare_one(df, cfg)

    # 종목명 / 섹터
    name = ""
    try:
        listing = get_listing(cfg)
        match = listing[listing["Code"].astype(str) == ticker]
        if not match.empty and "Name" in match.columns:
            name = str(match.iloc[0]["Name"])
    except Exception:
        pass
    sector = ""
    try:
        sec_map = get_sector_map(cfg)
        sector = sec_map.get(ticker, "")
    except Exception:
        pass

    last = prepared.iloc[-1]
    checks = _trend_template_checks(last)
    passes_all = bool(checks) and all(c["ok"] for c in checks)
    pass_count = sum(1 for c in checks if c["ok"])

    # Breakout 시그널 history
    qb = prepared["quality_breakout"].fillna(False).astype(bool) if "quality_breakout" in prepared.columns else pd.Series(False, index=prepared.index)
    pb = prepared["pivot_breakout"].fillna(False).astype(bool) if "pivot_breakout" in prepared.columns else pd.Series(False, index=prepared.index)
    sig_rows = prepared[qb | pb].copy()
    breakouts: list[dict] = []
    for dt, r in sig_rows.tail(max_breakouts).iterrows():
        kind = "quality" if bool(r.get("quality_breakout", False)) else "pivot"
        breakouts.append({
            "date": dt.strftime("%Y-%m-%d"),
            "type": kind,
            "close": _safe_float(r.get("Close")),
            "fwd_ret_5d": _safe_float(r.get("fwd_ret_5d")),
            "fwd_ret_20d": _safe_float(r.get("fwd_ret_20d")),
            "fwd_ret_60d": _safe_float(r.get("fwd_ret_60d")),
            "fwd_mdd_20d": _safe_float(r.get("fwd_mdd_20d")),
            "hit_20d": None if pd.isna(r.get("hit_20d")) else bool(r.get("hit_20d")),
        })
    breakouts.sort(key=lambda x: x["date"], reverse=True)

    # 현재 setup 플래그
    close = _safe_float(last.get("Close"))
    high52 = _safe_float(last.get("high_52w"))
    near_high = (close is not None and high52 is not None and
                 close >= high52 * (1.0 - cfg.near_high_pct))
    setup_flags = {
        "near_52w_high": near_high,
        "vcp_compression": _safe_bool(last.get("vcp_compression")),
        "volume_surge": _safe_bool(last.get("vol_surge")),
        "pivot_breakout": _safe_bool(last.get("pivot_breakout")),
        "quality_breakout": _safe_bool(last.get("quality_breakout")),
        "new_52w_high": _safe_bool(last.get("new_52w_high")),
        "template": _safe_bool(last.get("template")),
    }

    # Buy / Stop / Target (Minervini 룰: stop = -8% (or -7%), target ≥ +20%)
    pivot_series = prepared["High"].rolling(cfg.pivot_window).max()
    pivot_price = _safe_float(pivot_series.iloc[-1])
    atr_pct = _safe_float(last.get("atr_pct"))
    buy_setup = {
        "pivot_price": pivot_price,
        "current_close": close,
        "stop_8pct": (close * 0.92) if close is not None else None,
        "stop_atr": (close * (1.0 - 2.0 * atr_pct)) if (close is not None and atr_pct) else None,
        "target_10pct": (close * 1.10) if close is not None else None,
        "target_20pct": (close * 1.20) if close is not None else None,
        "atr_pct": atr_pct,
    }

    # 차트용 시계열 (최근 N거래일)
    tail = prepared.tail(chart_lookback)
    series = {
        "ohlc": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": _safe_float(r["Open"]),
                "high": _safe_float(r["High"]),
                "low": _safe_float(r["Low"]),
                "close": _safe_float(r["Close"]),
            } for d, r in tail.iterrows()
        ],
        "sma50": _series_pts(tail["SMA50"]) if "SMA50" in tail else [],
        "sma150": _series_pts(tail["SMA150"]) if "SMA150" in tail else [],
        "sma200": _series_pts(tail["SMA200"]) if "SMA200" in tail else [],
        "volume": _series_pts(tail["Volume"]),
        "high_52w": _series_pts(tail["high_52w"]) if "high_52w" in tail else [],
    }
    # 시그널 마커 (차트 윈도우 안만)
    chart_start = tail.index[0]
    marks: list[dict] = []
    for dt, r in sig_rows.iterrows():
        if dt < chart_start:
            continue
        marks.append({
            "date": dt.strftime("%Y-%m-%d"),
            "price": _safe_float(r.get("Close")),
            "type": "quality" if bool(r.get("quality_breakout", False)) else "pivot",
        })

    return {
        "ticker": ticker,
        "market": market,
        "name": name,
        "sector": sector,
        "as_of": tail.index[-1].strftime("%Y-%m-%d"),
        "n_bars": int(len(prepared)),
        "current": {
            "close": close,
            "volume": _safe_float(last.get("Volume")),
            "high_52w": high52,
            "low_52w": _safe_float(last.get("low_52w")),
            "pct_from_high": _safe_float(last.get("pct_from_high")),
            "pct_from_low": _safe_float(last.get("pct_from_low")),
            "atr_pct": atr_pct,
            "sma50": _safe_float(last.get("SMA50")),
            "sma150": _safe_float(last.get("SMA150")),
            "sma200": _safe_float(last.get("SMA200")),
            "sma200_slope": _safe_float(last.get("sma200_slope")),
            "ret_1m": _safe_float(last.get("ret_1m")),
            "ret_3m": _safe_float(last.get("ret_3m")),
            "ret_6m": _safe_float(last.get("ret_6m")),
            "ret_12m": _safe_float(last.get("ret_12m")),
            "ret_12_1m": _safe_float(last.get("ret_12_1m")),
            "rs_raw": _safe_float(last.get("rs_raw")),
        },
        "trend_template": {
            "passes_all": passes_all,
            "pass_count": pass_count,
            "total": len(checks),
            "checks": checks,
        },
        "breakouts": breakouts,
        "breakout_marks": marks,
        "setup_flags": setup_flags,
        "buy_setup": buy_setup,
        "series": series,
    }
