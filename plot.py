"""Leading indicator 시각화."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 호환
import matplotlib.pyplot as plt
import pandas as pd


def plot_dashboard(daily: pd.DataFrame, bo_rolling: pd.DataFrame | None,
                   factor_hits: pd.DataFrame | None, kospi_idx: pd.DataFrame | None,
                   save_path: str | Path) -> str:
    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)
    ax_idx, ax_breadth, ax_nh, ax_bo, ax_hit = axes

    if kospi_idx is not None and not kospi_idx.empty:
        ax_idx.plot(kospi_idx.index, kospi_idx["Close"], color="black", lw=1.2, label="KOSPI")
        ax_idx.set_ylabel("KOSPI")
        ax_idx.legend(loc="upper left")
        ax_idx.grid(alpha=0.3)
    else:
        ax_idx.set_visible(False)

    ax_breadth.plot(daily.index, daily["pct_above_sma200"] * 100, label="% > SMA200", color="navy")
    ax_breadth.plot(daily.index, daily["pct_above_sma50"] * 100, label="% > SMA50",
                    color="cornflowerblue", alpha=0.7)
    ax_breadth.plot(daily.index, daily["pct_passing_template"] * 100, label="% Trend Template",
                    color="darkgreen", alpha=0.8)
    ax_breadth.set_ylabel("Breadth (%)")
    ax_breadth.legend(loc="upper left")
    ax_breadth.grid(alpha=0.3)

    ax_nh.bar(daily.index, daily["new_highs"], color="green", alpha=0.55, label="신고가")
    ax_nh.bar(daily.index, -daily["new_lows"], color="red", alpha=0.55, label="신저가")
    ax_nh.axhline(0, color="black", lw=0.5)
    ax_nh.set_ylabel("Count")
    ax_nh.legend(loc="upper left")
    ax_nh.grid(alpha=0.3)

    ax_bo.plot(daily.index, daily["pivot_breakouts"], label="Pivot breakouts",
               color="steelblue", alpha=0.6)
    ax_bo.plot(daily.index, daily["quality_breakouts"], label="Quality breakouts",
               color="orangered", lw=1.5)
    ax_bo.set_ylabel("Breakouts / day")
    ax_bo.legend(loc="upper left")
    ax_bo.grid(alpha=0.3)

    if bo_rolling is not None and not bo_rolling.empty and "roll_hit_rate" in bo_rolling.columns:
        ax_hit.plot(bo_rolling.index, bo_rolling["roll_hit_rate"] * 100,
                    color="purple", lw=1.6, label="20d 브레이크아웃 hit rate (%)")
        ax_hit.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax_hit.set_ylabel("Hit rate (%)", color="purple")
        ax_hit.tick_params(axis="y", labelcolor="purple")
        ax_hit.grid(alpha=0.3)
        ax_hit.legend(loc="upper left")
        if factor_hits is not None and not factor_hits.empty and "hit_rate_pos" in factor_hits.columns:
            twin = ax_hit.twinx()
            twin.plot(factor_hits.index, factor_hits["hit_rate_pos"] * 100,
                      color="teal", alpha=0.6, label="모멘텀 데실 + ret hit rate (%)")
            twin.set_ylabel("Decile hit rate (%)", color="teal")
            twin.tick_params(axis="y", labelcolor="teal")
            twin.legend(loc="upper right")
    else:
        ax_hit.text(0.5, 0.5, "시그널 부족", ha="center", va="center",
                    transform=ax_hit.transAxes)

    fig.suptitle("KOSPI 모멘텀 / 브레이크아웃 Leading Indicator 대시보드",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)
