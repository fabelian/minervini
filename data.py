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

def get_kospi_listing(cache_dir: str = ".cache_kospi", refresh: bool = False) -> pd.DataFrame:
    cp = _cache_path("kospi_listing", cache_dir)
    if cp.exists() and not refresh:
        return pd.read_pickle(cp)

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
        raise RuntimeError("FinanceDataReader 또는 pykrx 가 필요합니다 (pip install -r requirements.txt).")

    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df.to_pickle(cp)
    return df


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
    if not _HAS_FDR:
        return {}
    try:
        df = fdr.StockListing("KOSPI")
    except Exception:
        return {}
    code_col = None
    for cand in ("Code", "Symbol", "code", "ticker"):
        if cand in df.columns:
            code_col = cand
            break
    sec_col = None
    for cand in ("Sector", "sector", "Industry", "업종"):
        if cand in df.columns:
            sec_col = cand
            break
    if not code_col or not sec_col:
        return {}
    out: dict[str, str] = {}
    for c, s in zip(df[code_col].astype(str).str.zfill(6), df[sec_col]):
        if pd.notna(s):
            sv = str(s).strip()
            if sv and sv.lower() != "nan":
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


def get_sector_map(cache_dir: str = ".cache_kospi", refresh: bool = False) -> dict[str, str]:
    """code → sector 라벨. FDR Sector 컬럼을 우선 사용하고 누락분만 pykrx로 보강."""
    cp = _cache_path("sector_map", cache_dir)
    if cp.exists() and not refresh:
        try:
            with cp.open("rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached:
                return cached
        except Exception:
            pass
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


def get_kospi_index(start: str, end: str, cache_dir: str = ".cache_kospi") -> pd.DataFrame | None:
    """KOSPI 종합지수 (KS11) - 비교용."""
    cp = _cache_path("index_kospi", cache_dir)
    if _HAS_FDR:
        try:
            df = fdr.DataReader("KS11", start, end)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                df.to_pickle(cp)
                return df
        except Exception:
            pass
    if cp.exists():
        return pd.read_pickle(cp)
    return None
