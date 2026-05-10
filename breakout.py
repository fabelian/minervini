"""브레이크아웃 탐지 로직.

- pivot_breakout: N거래일 신고가 돌파
- new_52w_high: 52주 신고가 갱신
- vol_surge: 50일 평균 거래량 대비 급증
- vcp_compression: 변동성 압축 (ATR/Price가 60거래일 분포 하위 분위)
- quality_breakout: pivot + vol_surge + 신고가 근방 + 트렌드 템플릿
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def detect_breakouts(df: pd.DataFrame, pivot_window: int = 50, vol_mult: float = 1.4,
                     near_high_pct: float = 0.05, vcp_q: float = 0.30) -> pd.DataFrame:
    out = df.copy()

    prior_pivot_high = out["Close"].shift(1).rolling(pivot_window).max()
    out["pivot_breakout"] = out["Close"] > prior_pivot_high

    avg_vol_50 = out["Volume"].rolling(50).mean()
    out["vol_surge"] = out["Volume"] > avg_vol_50 * vol_mult

    prior_high_252 = out["Close"].shift(1).rolling(252).max()
    out["new_52w_high"] = out["Close"] > prior_high_252

    out["atr20"] = _atr(out, 20)
    out["atr_pct"] = out["atr20"] / out["Close"]
    out["vcp_compression"] = (
        out["atr_pct"] < out["atr_pct"].rolling(60).quantile(vcp_q)
    )

    near_high = out["Close"] >= out.get("high_52w", out["Close"]) * (1.0 - near_high_pct)
    template = out["template"] if "template" in out.columns else True

    out["quality_breakout"] = (
        out["pivot_breakout"]
        & out["vol_surge"]
        & near_high
        & template
    )
    return out
