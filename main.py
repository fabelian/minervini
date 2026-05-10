"""KOSPI 모멘텀 / 브레이크아웃 모니터 CLI 진입점.

사용법:
    python main.py                       # 전체 KOSPI
    python main.py --max-stocks 50       # 빠른 테스트용 50종목
    python main.py --out-dir ./report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import Config
from monitor import run
from plot import plot_dashboard


def _print_summary(res: dict) -> None:
    leading = res.get("leading", pd.DataFrame())
    bo_rolling = res.get("breakout_rolling", pd.DataFrame())
    factor = res.get("factor_hit_rate", pd.DataFrame())
    picks = res.get("today_picks", pd.DataFrame())

    print("\n=== 요약 ===")
    if not leading.empty:
        last = leading.iloc[-1]
        print(f"기준일: {leading.index[-1].date()}")
        print(f"  종목 수 (집계 대상):     {len(res['prepared'])}")
        print(f"  % > SMA200 (시장 폭):    {last['pct_above_sma200']:.1%}")
        print(f"  % > SMA50:               {last['pct_above_sma50']:.1%}")
        print(f"  % Trend Template 통과:   {last['pct_passing_template']:.1%}")
        print(f"  52주 신고가:             {int(last['new_highs'])}")
        print(f"  52주 신저가:             {int(last['new_lows'])}")
        print(f"  Pivot 브레이크아웃:      {int(last['pivot_breakouts'])}")
        print(f"  Quality 브레이크아웃:    {int(last['quality_breakouts'])}")
        print(f"  평균 RS rank:            {last['avg_rs_rank']:.1f}")

    if not bo_rolling.empty and bo_rolling["roll_hit_rate"].dropna().size > 0:
        rr = bo_rolling.dropna(subset=["roll_hit_rate"]).iloc[-1]
        print("\n  [Quality 브레이크아웃 success rate (leading indicator)]")
        print(f"    20거래일 롤링 hit rate:   {rr['roll_hit_rate']:.1%}")
        print(f"    20거래일 평균 forward ret: {rr['roll_avg_ret']:+.2%}")
        print(f"    평균 max drawdown:         {rr['roll_avg_mdd']:+.2%}")

    if not factor.empty:
        last = factor.iloc[-1]
        print("\n  [모멘텀 상위 데실 hit rate (leading indicator)]")
        print(f"    상위 종목 수:              {int(last['n_top'])}")
        print(f"    >0% hit rate:              {last['hit_rate_pos']:.1%}")
        print(f"    >5% hit rate:              {last['hit_rate_5pct']:.1%}")
        print(f"    평균 forward ret:          {last['avg_ret']:+.2%}")

    rotation = res.get("rotation_score", pd.DataFrame())
    ranking = res.get("sector_ranking", pd.DataFrame())
    if not rotation.empty and "rotation_score" in rotation.columns:
        valid = rotation.dropna(subset=["rotation_score"])
        if not valid.empty:
            last = valid.iloc[-1]
            print("\n  [Sector rotation]")
            bs = last.get("bsr_spread", float("nan"))
            ms = last.get("mfhr_spread", float("nan"))
            print(f"    rotation_score (오늘):     {last['rotation_score']:.3f}"
                  f"  (BSR spread {bs:.3f} / MFHR spread {ms:.3f})")
    if not ranking.empty:
        print("    상위 5 섹터 (composite rank):")
        cols = [c for c in ["sector", "bsr", "mfhr", "composite_rank"] if c in ranking.columns]
        print(ranking[cols].head(5).to_string(index=False))
        print("    하위 3 섹터:")
        print(ranking[cols].tail(3).to_string(index=False))

    if not picks.empty:
        print(f"\n  [오늘의 Quality 브레이크아웃 종목 (RS≥{Config().pick_min_rs_rank:.0f}): "
              f"{len(picks)}개, 상위 10개]")
        cols = ["code", "name", "close", "rs_rank", "ret_3m", "pct_from_high", "vol_ratio"]
        cols = [c for c in cols if c in picks.columns]
        print(picks[cols].head(10).to_string(index=False))
    else:
        print("\n  오늘 Quality 브레이크아웃 종목 없음.")


def main() -> None:
    ap = argparse.ArgumentParser(description="KOSPI 모멘텀/브레이크아웃 leading indicator")
    ap.add_argument("--max-stocks", type=int, default=None,
                    help="유니버스를 N개로 제한 (테스트용)")
    ap.add_argument("--out-dir", default="output_kospi",
                    help="리포트 저장 디렉토리")
    ap.add_argument("--no-plot", action="store_true",
                    help="대시보드 PNG 생략")
    args = ap.parse_args()

    cfg = Config()
    res = run(cfg, max_stocks=args.max_stocks)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not res["leading"].empty:
        res["leading"].to_csv(out / "leading_indicators.csv")
    if not res["breakout_signals"].empty:
        res["breakout_signals"].to_csv(out / "breakout_signals.csv", index=False)
    if not res["breakout_rolling"].empty:
        res["breakout_rolling"].to_csv(out / "breakout_rolling_hit_rate.csv")
    if not res["pivot_signals"].empty:
        res["pivot_signals"].to_csv(out / "pivot_signals.csv", index=False)
    if not res["pivot_rolling"].empty:
        res["pivot_rolling"].to_csv(out / "pivot_rolling_hit_rate.csv")
    if not res["factor_hit_rate"].empty:
        res["factor_hit_rate"].to_csv(out / "factor_decile_hit_rate.csv")
    if not res.get("bsr_by_sector", pd.DataFrame()).empty:
        res["bsr_by_sector"].to_csv(out / "sector_breakout_hit_rate.csv", index=False)
    if not res.get("mfhr_by_sector", pd.DataFrame()).empty:
        res["mfhr_by_sector"].to_csv(out / "sector_factor_hit_rate.csv", index=False)
    if not res.get("rotation_score", pd.DataFrame()).empty:
        res["rotation_score"].to_csv(out / "rotation_score.csv")
    if not res.get("sector_ranking", pd.DataFrame()).empty:
        res["sector_ranking"].to_csv(out / "sector_ranking.csv", index=False)
    if not res["today_picks"].empty:
        res["today_picks"].to_csv(out / "today_picks.csv", index=False)

    if not args.no_plot and not res["leading"].empty:
        png = plot_dashboard(
            res["leading"],
            res["breakout_rolling"],
            res["factor_hit_rate"],
            res["kospi_index"],
            out / "dashboard.png",
        )
        print(f"\n대시보드 저장: {png}")

    _print_summary(res)
    print(f"\n리포트 디렉토리: {out.resolve()}")


if __name__ == "__main__":
    main()
