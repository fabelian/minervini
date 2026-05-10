"""오케스트레이션 + leading indicator 패널 빌드.

leading indicator는 횡단면 집계로 산출:
- 시장 폭(breadth): SMA50/200 위 비율, 트렌드 템플릿 통과 비율
- 신고가/신저가 갯수
- pivot/quality 브레이크아웃 갯수
- 최근 발생한 brake out 시그널의 롤링 hit rate
- 모멘텀 데실의 롤링 hit rate
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from data import get_kospi_listing, get_ohlcv_batch, get_kospi_index, get_sector_map
from factors import add_smas, momentum_returns, relative_strength, trend_template, cross_sectional_rank
from breakout import detect_breakouts
from hit_rate import (
    compute_forward_returns,
    label_hits_path,
    collect_signal_results,
    rolling_signal_hit_rate,
    factor_decile_hit_rate,
    rolling_signal_hit_rate_by_sector,
    factor_decile_hit_rate_by_sector,
    sector_rotation_score,
    sector_latest_ranking,
)


def _filter_universe(raw: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    keep: dict[str, pd.DataFrame] = {}
    for code, df in raw.items():
        if df is None or df.empty or len(df) < cfg.sma_long + 30:
            continue
        recent = df.tail(60)
        if recent.empty:
            continue
        last_close = recent["Close"].iloc[-1]
        if not np.isfinite(last_close) or last_close < cfg.min_price_krw:
            continue
        if recent["Volume"].mean() < cfg.min_avg_volume:
            continue
        keep[code] = df
    return keep


def _prepare_one(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_smas(df, (cfg.sma_short, cfg.sma_mid, cfg.sma_long))
    df = momentum_returns(df)
    df = relative_strength(df)
    df = trend_template(df)
    df = detect_breakouts(df, cfg.pivot_window, cfg.volume_surge_mult,
                          cfg.near_high_pct, cfg.vcp_atr_quantile)
    df = compute_forward_returns(df, cfg.forward_horizons)
    for h in cfg.forward_horizons:
        df = label_hits_path(df, h, cfg.success_threshold, cfg.stop_threshold)
    return df


def build_leading_indicator(prepared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not prepared:
        return pd.DataFrame()

    parts: dict[str, pd.DataFrame] = {}
    for code, df in prepared.items():
        d = pd.DataFrame(index=df.index)
        d["above_50"] = (df["Close"] > df["SMA50"]).astype(float)
        d["above_200"] = (df["Close"] > df["SMA200"]).astype(float)
        d["template"] = df["template"].astype(float)
        d["new_high"] = df["new_52w_high"].astype(float)
        # 신저가
        prior_low_252 = df["Close"].shift(1).rolling(252).min()
        d["new_low"] = (df["Close"] < prior_low_252).astype(float)
        d["pivot_bo"] = df["pivot_breakout"].astype(float)
        d["quality_bo"] = df["quality_breakout"].astype(float)
        d["rs_rank"] = df.get("rs_rank", pd.Series(np.nan, index=df.index))
        d["ret_3m"] = df["ret_3m"]
        parts[code] = d

    big = pd.concat(parts, axis=1)

    daily = pd.DataFrame(index=big.index)
    daily["pct_above_sma50"] = big.xs("above_50", level=1, axis=1).mean(axis=1, skipna=True)
    daily["pct_above_sma200"] = big.xs("above_200", level=1, axis=1).mean(axis=1, skipna=True)
    daily["pct_passing_template"] = big.xs("template", level=1, axis=1).mean(axis=1, skipna=True)
    daily["new_highs"] = big.xs("new_high", level=1, axis=1).sum(axis=1, skipna=True)
    daily["new_lows"] = big.xs("new_low", level=1, axis=1).sum(axis=1, skipna=True)
    daily["nh_minus_nl"] = daily["new_highs"] - daily["new_lows"]
    daily["pivot_breakouts"] = big.xs("pivot_bo", level=1, axis=1).sum(axis=1, skipna=True)
    daily["quality_breakouts"] = big.xs("quality_bo", level=1, axis=1).sum(axis=1, skipna=True)
    daily["avg_rs_rank"] = big.xs("rs_rank", level=1, axis=1).mean(axis=1, skipna=True)
    daily["median_ret_3m"] = big.xs("ret_3m", level=1, axis=1).median(axis=1, skipna=True)
    return daily


def todays_picks(prepared: dict[str, pd.DataFrame], cfg: Config,
                 listing: pd.DataFrame | None = None) -> pd.DataFrame:
    name_map: dict[str, str] = {}
    if listing is not None and "Code" in listing.columns and "Name" in listing.columns:
        name_map = dict(zip(listing["Code"].astype(str).str.zfill(6), listing["Name"]))
    rows = []
    for code, df in prepared.items():
        last = df.iloc[-1]
        if not bool(last.get("quality_breakout", False)):
            continue
        rs = float(last.get("rs_rank", 0.0) or 0.0)
        if rs < cfg.pick_min_rs_rank:
            continue
        rows.append({
            "code": code,
            "name": name_map.get(code, ""),
            "close": float(last["Close"]),
            "rs_rank": rs,
            "ret_3m": float(last.get("ret_3m", np.nan)),
            "ret_6m": float(last.get("ret_6m", np.nan)),
            "pct_from_high": float(last.get("pct_from_high", np.nan)),
            "atr_pct": float(last.get("atr_pct", np.nan)),
            "vol_ratio": float(last.get("Volume", np.nan)) /
                          max(float(df["Volume"].tail(50).mean()), 1.0),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rs_rank", ascending=False).reset_index(drop=True)


def run(cfg: Config | None = None, max_stocks: int | None = None) -> dict[str, Any]:
    cfg = cfg or Config()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=int(cfg.lookback_days * 1.6))).strftime("%Y-%m-%d")

    print(f"[1/6] KOSPI 종목 리스트 + 섹터 매핑 로딩 ({end})")
    listing = get_kospi_listing(cfg.cache_dir)
    sector_map = get_sector_map(cfg.cache_dir)
    print(f"      섹터 매핑: {len(sector_map)}종목 → {len(set(sector_map.values()))}섹터")
    codes = listing["Code"].astype(str).str.zfill(6).tolist()
    if max_stocks:
        codes = codes[:max_stocks]
    print(f"      대상 종목: {len(codes)}")

    print(f"[2/6] OHLCV 다운로드 (기간 {start}~{end}, 캐시 사용)")
    raw = get_ohlcv_batch(codes, start, end, cfg.cache_dir, workers=cfg.download_workers)
    print(f"      수집 종목: {len(raw)}")

    print("[3/6] 유동성 필터")
    raw = _filter_universe(raw, cfg)
    print(f"      필터 후: {len(raw)}")

    print("[4/6] 팩터 / 브레이크아웃 / forward return 계산")
    prepared: dict[str, pd.DataFrame] = {}
    for code, df in tqdm(raw.items(), desc="prep"):
        try:
            prepared[code] = _prepare_one(df, cfg)
        except Exception:
            continue

    print("[5/6] 횡단면 RS 랭크 부여")
    prepared = cross_sectional_rank(prepared, "rs_raw", "rs_rank")

    print("[6/6] Leading indicator 집계")
    leading = build_leading_indicator(prepared)

    bo_signals = collect_signal_results(prepared, "quality_breakout", cfg.hit_horizon)
    bo_rolling = rolling_signal_hit_rate(bo_signals, cfg.rolling_window)

    pivot_signals = collect_signal_results(prepared, "pivot_breakout", cfg.hit_horizon)
    pivot_rolling = rolling_signal_hit_rate(pivot_signals, cfg.rolling_window)

    factor_hits = factor_decile_hit_rate(
        prepared, score_col="rs_rank",
        top_pct=cfg.factor_top_pct,
        horizon=cfg.hit_horizon,
    )

    print("       Sector rotation 시그널 (BSR/MFHR by sector)")
    bsr_by_sector = rolling_signal_hit_rate_by_sector(
        bo_signals, sector_map, cfg.rolling_window, min_obs=10,
    )
    mfhr_by_sector = factor_decile_hit_rate_by_sector(
        prepared, sector_map, score_col="rs_rank",
        top_pct=cfg.factor_top_pct, horizon=cfg.hit_horizon, min_obs=10,
    )
    rotation = sector_rotation_score(bsr_by_sector, mfhr_by_sector)
    sector_ranking = sector_latest_ranking(bsr_by_sector, mfhr_by_sector)

    picks = todays_picks(prepared, cfg, listing)
    kospi_idx = get_kospi_index(start, end, cfg.cache_dir)

    return {
        "config": cfg,
        "listing": listing,
        "sector_map": sector_map,
        "kospi_index": kospi_idx,
        "prepared": prepared,
        "leading": leading,
        "breakout_signals": bo_signals,
        "breakout_rolling": bo_rolling,
        "pivot_signals": pivot_signals,
        "pivot_rolling": pivot_rolling,
        "factor_hit_rate": factor_hits,
        "bsr_by_sector": bsr_by_sector,
        "mfhr_by_sector": mfhr_by_sector,
        "rotation_score": rotation,
        "sector_ranking": sector_ranking,
        "today_picks": picks,
    }
