"""KOSPI / NASDAQ 모멘텀·브레이크아웃 모니터 설정값."""
from dataclasses import dataclass, field


@dataclass
class Config:
    market: str = "KOSPI"               # "KOSPI" | "NASDAQ"
    lookback_days: int = 504            # 약 2년 (252 거래일/년)

    # 유동성 필터 — 통화는 시장에 따른다 (KRW / USD)
    min_price: float = 2_000.0
    min_avg_volume: int = 50_000        # 최근 60거래일 평균 거래량

    # 이동평균 윈도우
    sma_short: int = 50
    sma_mid: int = 150
    sma_long: int = 200
    high_window: int = 252              # 52주

    # 브레이크아웃 파라미터
    pivot_window: int = 50              # N거래일 신고가 돌파
    volume_surge_mult: float = 1.4      # 거래량 급증 배수
    near_high_pct: float = 0.05         # 52주 신고가 ±5% 이내
    vcp_atr_quantile: float = 0.30      # ATR 비율이 60거래일 30분위 이하면 압축

    # 정방향 수익률 / hit 라벨링
    forward_horizons: tuple = (5, 20, 60)
    success_threshold: float = 0.10     # +10% 도달
    stop_threshold: float = -0.08       # -8% 손절 (Minervini 룰)
    hit_horizon: int = 20               # leading indicator용 기본 호라이즌

    # 팩터 hit rate
    factor_top_pct: float = 0.10        # 상위 10% 데실
    rolling_window: int = 20

    # 다운로드 동시성 / 캐시
    download_workers: int = 8
    cache_dir: str = ".cache_kospi"

    # 행동 가능한 종목 필터
    pick_min_rs_rank: float = 70.0

    @classmethod
    def for_market(cls, market: str) -> "Config":
        m = (market or "KOSPI").upper()
        if m == "NASDAQ":
            return cls(market="NASDAQ", min_price=5.0, cache_dir=".cache_nasdaq")
        return cls(market="KOSPI", min_price=2_000.0, cache_dir=".cache_kospi")
