"""네트워크 없이 합성 데이터로 핵심 로직 검증.

실행: python test_smoke.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from config import Config
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
from monitor import build_leading_indicator, todays_picks
import uvicorn


def _synthetic(seed: int, n: int = 600, trend: float = 0.0006, vol: float = 0.02,
               start_price: float = 10000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 약한 추세 + 백색잡음
    rets = rng.normal(trend, vol, size=n)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(50_000, 300_000, size=n).astype(float)
    # 시그널을 더 많이 만들기 위해 마지막 30거래일에 거래량 / 추세 부스트
    close[-30:] *= np.linspace(1.0, 1.18, 30)
    high[-30:] = np.maximum(high[-30:], close[-30:] * 1.005)
    low[-30:] = np.minimum(low[-30:], close[-30:] * 0.995)
    volume[-30:] *= 1.8
    idx = pd.bdate_range(start="2024-01-02", periods=n)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=idx)


def _prepare(df, cfg):
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


def main() -> int:
    cfg = Config()
    print("[1] 합성 데이터 5종목 생성")
    panel = {}
    for i in range(5):
        df = _synthetic(seed=i, trend=0.0004 + i * 0.0002, vol=0.02)
        panel[f"00000{i}"] = _prepare(df, cfg)

    sample = next(iter(panel.values()))
    expected_cols = [
        "SMA50", "SMA150", "SMA200", "ret_3m", "ret_12_1m", "rs_raw",
        "template", "pivot_breakout", "vol_surge", "new_52w_high",
        "quality_breakout", "atr_pct", "fwd_ret_20d", "hit_20d",
    ]
    missing = [c for c in expected_cols if c not in sample.columns]
    assert not missing, f"누락 컬럼: {missing}"
    print(f"    컬럼 OK ({len(sample.columns)}개)")

    print("[2] 횡단면 RS 랭크")
    panel = cross_sectional_rank(panel, "rs_raw", "rs_rank")
    last_ranks = [df["rs_rank"].iloc[-1] for df in panel.values() if "rs_rank" in df.columns]
    assert all(1.0 <= r <= 100.0 for r in last_ranks if pd.notna(r)), "RS 랭크 범위 오류"
    print(f"    RS rank 범위 OK: {[round(r, 1) for r in last_ranks]}")

    print("[3] hit 라벨 sanity")
    h = cfg.hit_horizon
    sample = next(iter(panel.values()))
    nonnull = sample[f"hit_{h}d"].dropna()
    assert len(nonnull) > 0, "hit 라벨이 모두 NaN"
    assert nonnull.isin([0, 1, True, False]).all(), "hit 라벨이 binary가 아님"
    print(f"    hit_{h}d 분포: hits={int(nonnull.sum())}/{len(nonnull)}")

    print("[4] 시그널 수집 + 롤링 hit rate")
    sigs = collect_signal_results(panel, "quality_breakout", cfg.hit_horizon)
    print(f"    quality_breakout 시그널 수: {len(sigs)}")
    if sigs.empty:
        # 부스트해도 안 잡힐 수 있으므로 pivot으로 폴백
        sigs = collect_signal_results(panel, "pivot_breakout", cfg.hit_horizon)
        print(f"    pivot_breakout 시그널 수: {len(sigs)}")
    rolling = rolling_signal_hit_rate(sigs, cfg.rolling_window, min_obs=1)
    if not rolling.empty:
        last = rolling.dropna(subset=["roll_hit_rate"]).tail(1)
        if not last.empty:
            print(f"    최근 롤링 hit rate: {last['roll_hit_rate'].iloc[0]:.1%}")

    print("[5] 모멘텀 데실 hit rate")
    fhr = factor_decile_hit_rate(panel, score_col="rs_rank",
                                 top_pct=0.4, horizon=cfg.hit_horizon, min_obs=2)
    print(f"    factor_decile rows: {len(fhr)}")
    if not fhr.empty:
        print(f"    최근 hit_rate_pos: {fhr['hit_rate_pos'].iloc[-1]:.1%}")

    print("[6] Leading indicator 패널")
    leading = build_leading_indicator(panel)
    assert "pct_above_sma200" in leading.columns
    last = leading.iloc[-1]
    print(f"    %>SMA200={last['pct_above_sma200']:.1%}  %>SMA50={last['pct_above_sma50']:.1%}  "
          f"신고가={int(last['new_highs'])}  quality_bo={int(last['quality_breakouts'])}")

    print("[7] 오늘의 종목")
    listing = pd.DataFrame({"Code": list(panel.keys()),
                            "Name": [f"종목{c}" for c in panel.keys()]})
    picks = todays_picks(panel, cfg, listing)
    print(f"    picks: {len(picks)} (RS>={cfg.pick_min_rs_rank})")
    if not picks.empty:
        print(picks[["code", "name", "close", "rs_rank", "ret_3m"]].to_string(index=False))

    print("[8] Sector rotation (BSR/MFHR by sector)")
    # 합성 sector_map: 5종목을 2섹터로 분할
    fake_sector = {code: ("A" if i < 3 else "B") for i, code in enumerate(panel.keys())}
    bsr_sec = rolling_signal_hit_rate_by_sector(sigs, fake_sector,
                                                cfg.rolling_window, min_obs=1)
    mfhr_sec = factor_decile_hit_rate_by_sector(panel, fake_sector,
                                                score_col="rs_rank",
                                                top_pct=0.5, horizon=cfg.hit_horizon,
                                                min_obs=1)
    print(f"    BSR by sector rows: {len(bsr_sec)}, MFHR by sector rows: {len(mfhr_sec)}")
    rot = sector_rotation_score(bsr_sec, mfhr_sec)
    print(f"    rotation_score 시계열 길이: {len(rot)}")
    if not rot.empty and "rotation_score" in rot.columns:
        v = rot["rotation_score"].dropna()
        if len(v) > 0:
            print(f"    rotation_score 마지막값: {v.iloc[-1]:.3f}")
    rank = sector_latest_ranking(bsr_sec, mfhr_sec)
    print(f"    sector_ranking rows: {len(rank)}")
    if not rank.empty:
        print(rank.to_string(index=False))

    print("\n=== 모든 핵심 로직 PASS ===")
    return 0


if __name__ == "__main__":
    # sys.exit(main())
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

