"""Forward return / 경로 의존적 hit 라벨 / 시그널 hit rate 집계.

핵심 아이디어:
1. 시그널 발생 시점 t에서 N거래일 forward 수익률, max gain, max drawdown 계산.
2. "Hit"는 N거래일 안에 success_threshold 도달이 stop_threshold 도달보다 먼저 일어난 경우.
   (Minervini: +10% 도달이 -8% 손절보다 먼저)
3. 일자별·시그널별 hit rate를 롤링 윈도우로 집계 → leading indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_returns(df: pd.DataFrame, horizons=(5, 20, 60)) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"fwd_ret_{h}d"] = out["Close"].shift(-h) / out["Close"] - 1.0
        out[f"fwd_max_{h}d"] = out["Close"].rolling(h).max().shift(-h) / out["Close"] - 1.0
        out[f"fwd_mdd_{h}d"] = out["Close"].rolling(h).min().shift(-h) / out["Close"] - 1.0
    return out


def label_hits_path(df: pd.DataFrame, horizon: int = 20,
                    success: float = 0.10, stop: float = -0.08,
                    out_col: str | None = None) -> pd.DataFrame:
    """경로 의존적 라벨링: 손절 트리거가 success보다 먼저 오면 미스, 아니면 hit."""
    out = df.copy()
    closes = out["Close"].values.astype(float)
    n = len(closes)
    hits = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)

    for i in range(n - 1):
        end_i = min(i + horizon, n - 1)
        if end_i <= i:
            continue
        base = closes[i]
        if not np.isfinite(base) or base <= 0:
            continue
        path = closes[i + 1: end_i + 1] / base - 1.0
        if len(path) == 0:
            continue
        valid[i] = True
        success_hits = np.where(path >= success)[0]
        stop_hits = np.where(path <= stop)[0]
        if len(success_hits) == 0:
            hits[i] = False
        elif len(stop_hits) == 0:
            hits[i] = True
        else:
            hits[i] = success_hits[0] < stop_hits[0]

    col = out_col or f"hit_{horizon}d"
    out[col] = np.where(valid, hits, np.nan)
    return out


def collect_signal_results(prepared: dict[str, pd.DataFrame], signal_col: str,
                           horizon: int = 20) -> pd.DataFrame:
    """모든 종목의 시그널 발생 일자에 대한 forward 결과를 long-format으로 수집."""
    rows = []
    fwd_col = f"fwd_ret_{horizon}d"
    hit_col = f"hit_{horizon}d"
    for code, df in prepared.items():
        if signal_col not in df.columns:
            continue
        sig = df[df[signal_col] == True]
        if sig.empty:
            continue
        for dt, row in sig.iterrows():
            rows.append({
                "date": dt,
                "code": code,
                "fwd_ret": row.get(fwd_col, np.nan),
                "fwd_max": row.get(f"fwd_max_{horizon}d", np.nan),
                "fwd_mdd": row.get(f"fwd_mdd_{horizon}d", np.nan),
                "hit": row.get(hit_col, np.nan),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "code", "fwd_ret", "fwd_max", "fwd_mdd", "hit"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def rolling_signal_hit_rate(signals: pd.DataFrame, window_days: int = 20,
                            min_obs: int = 10) -> pd.DataFrame:
    """일자별 hit rate를 롤링 윈도우(거래일이 아닌 캘린더 누적)로 평활."""
    if signals.empty:
        return pd.DataFrame()
    s = signals.dropna(subset=["hit"]).copy()
    if s.empty:
        return pd.DataFrame()
    daily = s.groupby("date").agg(
        n=("hit", "size"),
        hits=("hit", "sum"),
        avg_ret=("fwd_ret", "mean"),
        avg_mdd=("fwd_mdd", "mean"),
    )
    daily["roll_n"] = daily["n"].rolling(window_days, min_periods=1).sum()
    daily["roll_hits"] = daily["hits"].rolling(window_days, min_periods=1).sum()
    daily["roll_hit_rate"] = daily["roll_hits"] / daily["roll_n"].replace(0, np.nan)
    daily["roll_avg_ret"] = daily["avg_ret"].rolling(window_days, min_periods=1).mean()
    daily["roll_avg_mdd"] = daily["avg_mdd"].rolling(window_days, min_periods=1).mean()
    mask = daily["roll_n"] < min_obs
    daily.loc[mask, ["roll_hit_rate", "roll_avg_ret", "roll_avg_mdd"]] = np.nan
    return daily


def factor_decile_hit_rate(prepared: dict[str, pd.DataFrame], score_col: str = "rs_rank",
                           top_pct: float = 0.10, horizon: int = 20,
                           min_obs: int = 20) -> pd.DataFrame:
    """일자별로 score 상위 top_pct 분위 종목의 forward return 통계 (hit = ret>0)."""
    fwd_col = f"fwd_ret_{horizon}d"
    score_panel = pd.DataFrame({c: d[score_col] for c, d in prepared.items() if score_col in d.columns})
    fwd_panel = pd.DataFrame({c: d[fwd_col] for c, d in prepared.items() if fwd_col in d.columns})
    if score_panel.empty or fwd_panel.empty:
        return pd.DataFrame()

    common = score_panel.index.intersection(fwd_panel.index)
    score_panel = score_panel.loc[common]
    fwd_panel = fwd_panel.loc[common]

    rows = []
    for date in score_panel.index:
        s = score_panel.loc[date].dropna()
        if len(s) < min_obs:
            continue
        threshold = s.quantile(1.0 - top_pct)
        top_codes = s[s >= threshold].index
        f = fwd_panel.loc[date, top_codes].dropna()
        if f.empty:
            continue
        rows.append({
            "date": date,
            "n_top": int(len(top_codes)),
            "n_with_fwd": int(len(f)),
            "hit_rate_pos": float((f > 0).mean()),
            "hit_rate_5pct": float((f > 0.05).mean()),
            "avg_ret": float(f.mean()),
            "median_ret": float(f.median()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()
