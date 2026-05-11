"""KOSPI 종목 리스트 및 OHLCV 데이터 로더.

FinanceDataReader를 우선 사용하고, 미설치 시 pykrx로 폴백.
캐시는 종목별 pickle. 마지막 거래일이 요청 종료일 -3거래일보다 이전이면 재다운로드.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import pickle

import pandas as pd
from tqdm import tqdm

try:
    import FinanceDataReader as fdr
    _HAS_FDR = True
except Exception:
    _HAS_FDR = False

try:
    from pykrx import stock as krx
    _HAS_KRX = True
except Exception:
    _HAS_KRX = False


def _ensure_cache(cache_dir: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_path(name: str, cache_dir: str) -> Path:
    return _ensure_cache(cache_dir) / f"{name}.pkl"


# ---------- 종목 리스트 ----------

def get_listing(cfg, refresh: bool = False) -> pd.DataFrame:
    """시장별 종목 리스트. 'Code' 컬럼을 표준 식별자로 정규화한다."""
    market = (cfg.market or "KOSPI").upper()
    cache_dir = cfg.cache_dir
    cp = _cache_path(f"{market.lower()}_listing", cache_dir)
    if cp.exists() and not refresh:
        return pd.read_pickle(cp)

    if market == "KOSPI":
        if _HAS_FDR:
            df = fdr.StockListing("KOSPI")
            if "Code" not in df.columns:
                for cand in ("Symbol", "code", "ticker"):
                    if cand in df.columns:
                        df = df.rename(columns={cand: "Code"})
                        break
        elif _HAS_KRX:
            today = datetime.now().strftime("%Y%m%d")
            tickers = krx.get_market_ticker_list(today, market="KOSPI")
            df = pd.DataFrame({
                "Code": tickers,
                "Name": [krx.get_market_ticker_name(t) for t in tickers],
            })
        else:
            raise RuntimeError("FinanceDataReader 또는 pykrx 가 필요합니다.")
        df["Code"] = df["Code"].astype(str).str.zfill(6)
    elif market == "NASDAQ":
        if not _HAS_FDR:
            raise RuntimeError("NASDAQ 종목 리스트는 FinanceDataReader가 필요합니다.")
        df = fdr.StockListing("NASDAQ")
        if "Symbol" in df.columns and "Code" not in df.columns:
            df = df.rename(columns={"Symbol": "Code"})
        df["Code"] = df["Code"].astype(str)
    else:
        raise ValueError(f"unknown market: {market}")

    df.to_pickle(cp)
    return df


def get_kospi_listing(cache_dir: str = ".cache_kospi", refresh: bool = False) -> pd.DataFrame:
    """하위호환 wrapper — 신규 코드는 get_listing(cfg)를 사용."""
    from config import Config
    return get_listing(Config(market="KOSPI", cache_dir=cache_dir), refresh)


# ---------- 개별 OHLCV ----------

def _fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame | None:
    if _HAS_FDR:
        df = fdr.DataReader(code, start, end)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    if _HAS_KRX:
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")
        df = krx.get_market_ohlcv_by_date(s, e, code)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "시가": "Open", "고가": "High", "저가": "Low",
            "종가": "Close", "거래량": "Volume", "거래대금": "Value",
            "등락률": "Change",
        })
        df.index = pd.to_datetime(df.index)
        return df
    raise RuntimeError("데이터 라이브러리가 설치되어 있지 않습니다.")


def get_ohlcv(code: str, start: str, end: str, cache_dir: str = ".cache_kospi",
              refresh: bool = False) -> pd.DataFrame | None:
    cp = _cache_path(f"ohlcv_{code}", cache_dir)
    end_ts = pd.Timestamp(end)
    start_ts = pd.Timestamp(start)

    if cp.exists() and not refresh:
        try:
            df = pd.read_pickle(cp)
            if df is not None and not df.empty:
                covers_start = df.index.min() <= start_ts
                covers_end = df.index.max() >= end_ts - pd.Timedelta(days=4)
                if covers_start and covers_end:
                    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        except Exception:
            pass  # 손상된 캐시는 재다운로드

    df = _fetch_ohlcv(code, start, end)
    if df is None or df.empty:
        return None
    df.to_pickle(cp)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def get_ohlcv_batch(codes: list[str], start: str, end: str, cache_dir: str = ".cache_kospi",
                    workers: int = 8) -> dict[str, pd.DataFrame]:
    """병렬 다운로드. 캐시 hit이면 즉시 반환."""
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(get_ohlcv, c, start, end, cache_dir): c for c in codes}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="OHLCV"):
            code = futures[fut]
            try:
                df = fut.result()
                if df is not None and len(df) > 0:
                    out[code] = df
            except Exception:
                continue
    return out


# ---------- 섹터 매핑 ----------

def _fdr_sector_map() -> dict[str, str]:
    """FDR에서 KOSPI 종목별 섹터 매핑을 구한다.

    FDR `StockListing("KOSPI")`에는 Sector/Industry 컬럼이 없다.
    `StockListing("KRX-DESC")`가 Sector/Industry 컬럼을 제공하므로 그쪽을 사용한다.
    KOSPI 종목은 Sector가 거의 비어있고 Industry만 채워져 있어 Industry를 우선 사용한다."""
    if not _HAS_FDR:
        return {}
    df = None
    for listing in ("KRX-DESC", "KOSPI"):
        try:
            cand = fdr.StockListing(listing)
        except Exception:
            continue
        if cand is None or cand.empty:
            continue
        cols = set(cand.columns)
        if "Sector" in cols or "Industry" in cols or "업종" in cols:
            df = cand
            break
    if df is None:
        return {}

    code_col = None
    for cand in ("Code", "Symbol", "code", "ticker"):
        if cand in df.columns:
            code_col = cand
            break
    if not code_col:
        return {}

    # Market 컬럼이 있으면 KOSPI만 필터
    if "Market" in df.columns:
        df = df[df["Market"].astype(str).str.upper() == "KOSPI"]

    # Industry를 우선 (KOSPI 커버리지 ↑), 비어있으면 Sector로 폴백
    primary = next((c for c in ("Industry", "Sector", "sector", "업종") if c in df.columns), None)
    secondary = next((c for c in ("Sector", "Industry", "sector", "업종")
                      if c in df.columns and c != primary), None)
    if primary is None:
        return {}

    out: dict[str, str] = {}
    codes = df[code_col].astype(str).str.zfill(6)
    p_vals = df[primary]
    s_vals = df[secondary] if secondary else [None] * len(df)
    for c, p, s in zip(codes, p_vals, s_vals):
        sv = None
        for cand in (p, s):
            if cand is None:
                continue
            if pd.isna(cand):
                continue
            text = str(cand).strip()
            if text and text.lower() != "nan":
                sv = text
                break
        if sv:
            out[c] = sv
    return out


def _krx_sector_map() -> dict[str, str]:
    """KRX KOSPI 산업별 지수(1010~1090 등) 멤버를 통해 섹터 부여."""
    if not _HAS_KRX:
        return {}
    out: dict[str, str] = {}
    try:
        idx_codes = krx.get_index_ticker_list(market="KOSPI")
    except Exception:
        return {}
    for idx_code in idx_codes or []:
        if not (isinstance(idx_code, str) and idx_code.startswith("1") and len(idx_code) == 4):
            continue
        if idx_code == "1001":  # 종합지수
            continue
        try:
            name = krx.get_index_ticker_name(idx_code)
        except Exception:
            continue
        if not name:
            continue
        # 산업분류가 아닌 사이즈/스타일/파생 지수 제외
        if any(k in name for k in ("KOSPI 200", "코스피200", "선물", "변동성",
                                   "TOP", "대형", "중형", "소형", "고배당", "배당성장")):
            continue
        try:
            members = krx.get_index_portfolio_deposit_file(idx_code)
        except Exception:
            continue
        for code in members or []:
            out.setdefault(str(code).zfill(6), name)
    return out


def _fdr_sector_map_nasdaq() -> dict[str, str]:
    """NASDAQ — fdr.StockListing('NASDAQ')의 Industry 컬럼은 100% 채워져 있다."""
    if not _HAS_FDR:
        return {}
    try:
        df = fdr.StockListing("NASDAQ")
    except Exception:
        return {}
    if "Symbol" not in df.columns or "Industry" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for sym, ind in zip(df["Symbol"].astype(str), df["Industry"]):
        if pd.isna(ind):
            continue
        text = str(ind).strip()
        if text and text.lower() != "nan":
            out[sym] = text
    return out


def get_sector_map(cfg=None, refresh: bool = False, *,
                   cache_dir: str | None = None,
                   market: str | None = None) -> dict[str, str]:
    """code → sector 라벨.
    호출 방식 (둘 다 지원):
      - 신규: get_sector_map(cfg)
      - 구 호환: get_sector_map(cache_dir=..., market="KOSPI")
    """
    if cfg is not None:
        market = (cfg.market or "KOSPI").upper()
        cache_dir = cfg.cache_dir
    else:
        market = (market or "KOSPI").upper()
        cache_dir = cache_dir or ".cache_kospi"

    cp = _cache_path(f"sector_map_{market.lower()}", cache_dir)
    if cp.exists() and not refresh:
        try:
            with cp.open("rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached:
                return cached
        except Exception:
            pass

    if market == "NASDAQ":
        out = _fdr_sector_map_nasdaq()
    else:
        out = _fdr_sector_map()
        krx_map = _krx_sector_map()
        if not out:
            out = krx_map
        else:
            for code, sec in krx_map.items():
                out.setdefault(code, sec)

    if out:
        with cp.open("wb") as f:
            pickle.dump(out, f)
    return out


_MARKET_INDEX_SYMBOL = {"KOSPI": "KS11", "NASDAQ": "IXIC"}


def get_index(cfg, start: str, end: str) -> pd.DataFrame | None:
    """시장 벤치마크 인덱스 OHLC (KOSPI=KS11, NASDAQ=IXIC)."""
    market = (cfg.market or "KOSPI").upper()
    cache_dir = cfg.cache_dir
    sym = _MARKET_INDEX_SYMBOL.get(market)
    if not sym:
        return None
    cp = _cache_path(f"index_{market.lower()}", cache_dir)
    if _HAS_FDR:
        try:
            df = fdr.DataReader(sym, start, end)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                df.to_pickle(cp)
                return df
        except Exception:
            pass
    if cp.exists():
        return pd.read_pickle(cp)
    return None


def get_kospi_index(start: str, end: str, cache_dir: str = ".cache_kospi") -> pd.DataFrame | None:
    """하위호환 wrapper."""
    from config import Config
    return get_index(Config(market="KOSPI", cache_dir=cache_dir), start, end)
