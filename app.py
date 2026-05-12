"""FastAPI 백엔드: 파이프라인 트리거 + 결과 시리즈 노출 + 정적 HTML 서빙."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import db
from config import Config
from monitor import run as run_pipeline
from plot import plot_dashboard
from chat_agent import ChatAgent, DashboardContext
from single_analysis import analyze_ticker


app = FastAPI(title="KOSPI / NASDAQ Momentum Monitor", version="0.3.0")
db.init()  # DATABASE_URL이 있으면 캐시 활성, 없으면 no-op

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
SUPPORTED_MARKETS = ("KOSPI", "NASDAQ")
OUTPUT_DIRS: dict[str, Path] = {m: ROOT / f"output_{m.lower()}" for m in SUPPORTED_MARKETS}
for d in OUTPUT_DIRS.values():
    d.mkdir(exist_ok=True)
# 하위호환: 기존 코드 일부가 OUTPUT_DIR을 참조하는 경우를 위해 default = KOSPI
OUTPUT_DIR = OUTPUT_DIRS["KOSPI"]

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()
LATEST_BY_MARKET: dict[str, dict[str, Any]] = {}


def _resolve_market(value: str | None) -> str:
    m = (value or "KOSPI").upper()
    if m not in OUTPUT_DIRS:
        raise HTTPException(400, f"unsupported market: {m}")
    return m


def _client_ip(request: Request | None) -> str | None:
    """Railway/Reverse-proxy 뒤에서도 원본 IP를 가져온다.
    X-Forwarded-For 첫 항목 → X-Real-IP → request.client.host 순.
    """
    if request is None:
        return None
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip() or None
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip() or None
    if request.client and request.client.host:
        return request.client.host
    return None


# ---------- 직렬화 ----------

def _series(df: pd.DataFrame | None, col: str) -> list[dict]:
    if df is None or df.empty or col not in df.columns:
        return []
    s = df[col].dropna()
    return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in s.items()]


def _safe_records(df: pd.DataFrame | None) -> list[dict]:
    """NaN/inf를 None으로 변환해 JSON-safe records 반환."""
    if df is None or df.empty:
        return []
    out: list[dict] = []
    for _, row in df.iterrows():
        rec: dict = {}
        for k, v in row.items():
            if isinstance(v, float):
                if not (v == v) or v in (float("inf"), float("-inf")):  # NaN / inf
                    rec[k] = None
                else:
                    rec[k] = float(v)
            elif pd.isna(v):
                rec[k] = None
            else:
                rec[k] = v
        out.append(rec)
    return out


def _summarize(res: dict[str, Any]) -> dict[str, Any]:
    leading = res["leading"]
    bo_rolling = res["breakout_rolling"]
    factor = res["factor_hit_rate"]
    picks = res["today_picks"]
    bench = res.get("benchmark_index")
    if bench is None:
        bench = res.get("kospi_index")
    rotation = res.get("rotation_score")
    sector_ranking = res.get("sector_ranking")
    cfg_obj = res.get("config")
    market = getattr(cfg_obj, "market", "KOSPI") if cfg_obj is not None else "KOSPI"

    summary: dict[str, Any] = {
        "market": market,
        "n_stocks": len(res["prepared"]),
        "as_of": leading.index[-1].strftime("%Y-%m-%d") if not leading.empty else None,
        "latest": {},
        "series": {},
        "picks": [],
    }

    if not leading.empty:
        last = leading.iloc[-1]
        summary["latest"].update({
            "pct_above_sma200": float(last["pct_above_sma200"]),
            "pct_above_sma50": float(last["pct_above_sma50"]),
            "pct_passing_template": float(last["pct_passing_template"]),
            "new_highs": int(last["new_highs"]),
            "new_lows": int(last["new_lows"]),
            "pivot_breakouts": int(last["pivot_breakouts"]),
            "quality_breakouts": int(last["quality_breakouts"]),
            "avg_rs_rank": float(last["avg_rs_rank"]),
        })
        summary["series"].update({
            "pct_above_sma200": _series(leading, "pct_above_sma200"),
            "pct_above_sma50": _series(leading, "pct_above_sma50"),
            "pct_passing_template": _series(leading, "pct_passing_template"),
            "new_highs": _series(leading, "new_highs"),
            "new_lows": _series(leading, "new_lows"),
            "quality_breakouts": _series(leading, "quality_breakouts"),
            "pivot_breakouts": _series(leading, "pivot_breakouts"),
        })

    if bench is not None and not bench.empty:
        summary["series"]["kospi"] = _series(bench, "Close")  # 클라이언트는 시장 무관하게 'kospi' 키로 KOSPI/IXIC 시리즈를 표시

    if not bo_rolling.empty:
        summary["series"]["bo_hit_rate"] = _series(bo_rolling, "roll_hit_rate")
        summary["series"]["bo_avg_ret"] = _series(bo_rolling, "roll_avg_ret")
        last_hr = bo_rolling.dropna(subset=["roll_hit_rate"])
        if not last_hr.empty:
            summary["latest"]["bo_hit_rate"] = float(last_hr["roll_hit_rate"].iloc[-1])
            summary["latest"]["bo_avg_ret"] = float(last_hr["roll_avg_ret"].iloc[-1])

    if not factor.empty:
        summary["series"]["factor_hit_rate_pos"] = _series(factor, "hit_rate_pos")
        summary["series"]["factor_hit_rate_5pct"] = _series(factor, "hit_rate_5pct")
        summary["series"]["factor_avg_ret"] = _series(factor, "avg_ret")
        summary["latest"]["factor_hit_rate"] = float(factor["hit_rate_pos"].iloc[-1])
        summary["latest"]["factor_avg_ret"] = float(factor["avg_ret"].iloc[-1])

    if rotation is not None and not rotation.empty:
        if "rotation_score" in rotation.columns:
            summary["series"]["rotation_score"] = _series(rotation, "rotation_score")
            valid = rotation.dropna(subset=["rotation_score"])
            if not valid.empty:
                last = valid.iloc[-1]
                summary["latest"]["rotation_score"] = float(last["rotation_score"])
                if "bsr_spread" in valid.columns and pd.notna(last.get("bsr_spread")):
                    summary["latest"]["bsr_spread"] = float(last["bsr_spread"])
                if "mfhr_spread" in valid.columns and pd.notna(last.get("mfhr_spread")):
                    summary["latest"]["mfhr_spread"] = float(last["mfhr_spread"])
        if "bsr_spread" in rotation.columns:
            summary["series"]["bsr_spread"] = _series(rotation, "bsr_spread")
        if "mfhr_spread" in rotation.columns:
            summary["series"]["mfhr_spread"] = _series(rotation, "mfhr_spread")

    if sector_ranking is not None and not sector_ranking.empty:
        summary["sector_ranking"] = _safe_records(sector_ranking)
        first = sector_ranking.iloc[0]
        last_row = sector_ranking.iloc[-1]
        summary["latest"]["top_sector"] = str(first["sector"])
        summary["latest"]["bottom_sector"] = str(last_row["sector"])

    if not picks.empty:
        summary["picks"] = picks.to_dict(orient="records")
    return summary


# ---------- 백그라운드 잡 ----------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _execute(job_id: str, max_stocks: int | None, as_of: str | None,
             market: str, force_refresh: bool = False,
             client_ip: str | None = None) -> None:
    started_at = datetime.now(timezone.utc)
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = started_at.isoformat(timespec="seconds")
    try:
        cfg = Config.for_market(market)
        out_dir = OUTPUT_DIRS[market]
        cfg_hash = db.config_hash(cfg)

        # 캐시 조회 (force_refresh가 아니면)
        if not force_refresh:
            cached = db.lookup(market, as_of, max_stocks, cfg_hash)
            if cached:
                summary = cached["summary"]
                png_bytes = cached.get("dashboard_png")
                if png_bytes:
                    try:
                        (out_dir / "dashboard.png").write_bytes(png_bytes)
                    except Exception:
                        pass
                finished_at = datetime.now(timezone.utc)
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "completed"
                    JOBS[job_id]["finished_at"] = finished_at.isoformat(timespec="seconds")
                    JOBS[job_id]["summary"] = summary
                    JOBS[job_id]["source"] = "cache"
                LATEST_BY_MARKET[market] = {
                    "job_id": job_id, "ts": _now_iso(),
                    "source": "cache", "summary": summary,
                }
                return

        # 캐시 미스 → 실제 계산
        res = run_pipeline(cfg, max_stocks=max_stocks, as_of=as_of)
        png_bytes = None
        if not res["leading"].empty:
            bench = res.get("benchmark_index")
            if bench is None:
                bench = res.get("kospi_index")
            png_path = out_dir / "dashboard.png"
            plot_dashboard(
                res["leading"],
                res.get("breakout_rolling"),
                res.get("factor_hit_rate"),
                bench,
                png_path,
                market=market,
            )
            try:
                png_bytes = png_path.read_bytes()
            except Exception:
                png_bytes = None
        summary = _summarize(res)
        finished_at = datetime.now(timezone.utc)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["finished_at"] = finished_at.isoformat(timespec="seconds")
            JOBS[job_id]["summary"] = summary
            JOBS[job_id]["source"] = "fresh"
        LATEST_BY_MARKET[market] = {
            "job_id": job_id, "ts": _now_iso(),
            "source": "fresh", "summary": summary,
        }
        # DB 저장 (best-effort)
        try:
            cfg_dict = asdict(cfg)
            db.save(market, as_of, max_stocks, cfg_hash,
                    cfg_dict, summary, png_bytes,
                    int(summary.get("n_stocks") or 0),
                    started_at, finished_at,
                    client_ip=client_ip)
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = f"{type(e).__name__}: {e}"
            JOBS[job_id]["finished_at"] = _now_iso()


# ---------- API ----------

class RunRequest(BaseModel):
    max_stocks: int | None = None
    as_of: str | None = None
    market: str | None = "KOSPI"
    force_refresh: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        raise HTTPException(500, "static/index.html missing")
    return p.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "ts": _now_iso(), "db": db.status()}


@app.post("/api/run")
def post_run(req: RunRequest, request: Request) -> dict:
    market = _resolve_market(req.market)
    ip = _client_ip(request)
    job_id = uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "market": market,
            "max_stocks": req.max_stocks,
            "as_of": req.as_of,
            "force_refresh": req.force_refresh,
            "client_ip": ip,
            "queued_at": _now_iso(),
        }
    Thread(target=_execute,
           args=(job_id, req.max_stocks, req.as_of, market,
                 req.force_refresh, ip),
           daemon=True).start()
    return {"job_id": job_id, "status": "queued", "market": market,
            "cache_enabled": db.is_enabled()}


@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in job.items() if k != "summary"}


@app.get("/api/result/{job_id}")
def get_result(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "completed":
        raise HTTPException(409, f"job not ready (status={job['status']})")
    return job["summary"]


@app.get("/api/latest")
def latest(market: str = "KOSPI") -> dict:
    m = _resolve_market(market)
    rec = LATEST_BY_MARKET.get(m)
    if not rec:
        raise HTTPException(404, f"no result yet for {m}")
    return rec


@app.get("/api/dashboard.png")
def dashboard_png(market: str = "KOSPI"):
    m = _resolve_market(market)
    p = OUTPUT_DIRS[m] / "dashboard.png"
    if not p.exists():
        raise HTTPException(404, f"dashboard not generated yet for {m}")
    return FileResponse(p, media_type="image/png")


# ---------- 단일 종목 분석 ----------

class SingleRequest(BaseModel):
    ticker: str
    market: str | None = None
    as_of: str | None = None


@app.post("/api/single")
def post_single(req: SingleRequest) -> dict:
    try:
        return analyze_ticker(req.ticker, req.market, req.as_of)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ---------- Chat ----------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    provider: str | None = None  # "anthropic" | "openrouter"
    market: str | None = "KOSPI"


@app.post("/api/chat")
def post_chat(req: ChatRequest) -> dict:
    provider = (req.provider or "anthropic").lower()
    market = _resolve_market(req.market)
    rec = LATEST_BY_MARKET.get(market) or {}
    summary = rec.get("summary") or {}
    ctx = DashboardContext(latest_summary=summary, output_dir=OUTPUT_DIRS[market])
    try:
        agent = ChatAgent(model=req.model, provider=provider, context=ctx)
        reply = agent.chat([m.model_dump() for m in req.messages])
        return {"reply": reply, "model": agent.model, "provider": provider,
                "market": market}
    except EnvironmentError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
