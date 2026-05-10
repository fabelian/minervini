"""모멘텀 / 그로스 (가격 기반) 팩터 계산.

가격 베이스 팩터만 다룬다 (재무 그로스는 별도 데이터 소스 필요).
- 단/중/장기 모멘텀 (1/3/6/12개월, 12-1)
- IBD 스타일 RS 컴포지트 (이후 횡단면 퍼센타일 랭크)
- Minervini 트렌드 템플릿 플래그
- 52주 신고가 거리 / 52주 저점 거리
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_smas(df: pd.DataFrame, windows=(50, 150, 200)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out


def momentum_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["ret_1m"] = c.pct_change(21)
    out["ret_3m"] = c.pct_change(63)
    out["ret_6m"] = c.pct_change(126)
    out["ret_12m"] = c.pct_change(252)
    out["ret_12_1m"] = (c.shift(21) / c.shift(252)) - 1.0
    return out


def relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """IBD 가중치 (40/20/20/20) - 이후 횡단면 퍼센타일 랭크로 변환."""
    out = df.copy()
    c = out["Close"]
    q1 = c.pct_change(63)                    # 최근 분기
    q2 = c.shift(63).pct_change(63)
    q3 = c.shift(126).pct_change(63)
    q4 = c.shift(189).pct_change(63)
    out["rs_raw"] = 0.4 * q1 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4
    return out


def trend_template(df: pd.DataFrame) -> pd.DataFrame:
    """Mark Minervini 트렌드 템플릿. SMA50/150/200 컬럼이 있어야 한다."""
    out = df.copy()
    out["high_52w"] = out["Close"].rolling(252).max()
    out["low_52w"] = out["Close"].rolling(252).min()
    out["pct_from_high"] = out["Close"] / out["high_52w"] - 1.0
    out["pct_from_low"] = out["Close"] / out["low_52w"] - 1.0
    out["sma200_slope"] = out["SMA200"].diff(21)  # 200MA의 1개월 변화

    out["template"] = (
        (out["Close"] > out["SMA150"]) &
        (out["Close"] > out["SMA200"]) &
        (out["SMA150"] > out["SMA200"]) &
        (out["sma200_slope"] > 0) &
        (out["SMA50"] > out["SMA150"]) &
        (out["SMA50"] > out["SMA200"]) &
        (out["Close"] > out["SMA50"]) &
        (out["Close"] >= out["low_52w"] * 1.30) &      # 저점 대비 +30% 이상
        (out["Close"] >= out["high_52w"] * 0.75)       # 신고가 대비 -25% 이내
    )
    return out


def cross_sectional_rank(panel_dict: dict[str, pd.DataFrame], col: str = "rs_raw",
                         out_col: str = "rs_rank") -> dict[str, pd.DataFrame]:
    """모든 종목에 대해 일자별 횡단면 퍼센타일 랭크 (1~99) 부여.

    panel_dict는 {code: df} 형태. df에 col 컬럼이 있어야 한다.
    원본 dict를 in-place로 수정하고 반환한다.
    """
    cols = {c: d[col] for c, d in panel_dict.items() if col in d.columns}
    if not cols:
        return panel_dict
    panel = pd.DataFrame(cols)
    rank = panel.rank(axis=1, pct=True) * 99.0 + 1.0   # 1..100
    for code in panel_dict:
        if code in rank.columns:
            panel_dict[code][out_col] = rank[code]
    return panel_dict
