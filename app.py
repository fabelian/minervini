"""FastAPI 백엔드: 파이프라인 트리거 + 결과 시리즈 노출 + 정적 HTML 서빙."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import Config
from monitor import run as run_pipeline
from plot import plot_dashboard


app = FastAPI(title="KOSPI Momentum Monitor", version="0.1.0")

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "output_kospi"
OUTPUT_DIR.mkdir(exist_ok=True)

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()
LATEST: dict[str, Any] = {}


# ---------- 직렬화 ----------

def _series(df: pd.DataFrame | None, col: str) -> list[dict]:
    if df is None or df.empty or col not in df.columns:
        return []
    s = df[col].dropna()
    return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in s.items()]


def _summarize(res: dict[str, Any]) -> dict[str, Any]:
    leading = res["leading"]
    bo_rolling = res["breakout_rolling"]
    factor = res["factor_hit_rate"]
    picks = res["today_picks"]
    kospi = res.get("kospi_index")

    summary: dict[str, Any] = {
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

    if kospi is not None and not kospi.empty:
        summary["series"]["kospi"] = _series(kospi, "Close")

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

    if not picks.empty:
        summary["picks"] = picks.to_dict(orient="records")
    return summary


# ---------- 백그라운드 잡 ----------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _execute(job_id: str, max_stocks: int | None) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = _now_iso()
    try:
        cfg = Config()
        res = run_pipeline(cfg, max_stocks=max_stocks)
        if not res["leading"].empty:
            plot_dashboard(
                res["leading"],
                res.get("breakout_rolling"),
                res.get("factor_hit_rate"),
                res.get("kospi_index"),
                OUTPUT_DIR / "dashboard.png",
            )
        summary = _summarize(res)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["finished_at"] = _now_iso()
            JOBS[job_id]["summary"] = summary
        LATEST.update({"job_id": job_id, "ts": _now_iso(), "summary": summary})
    except Exception as e:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = f"{type(e).__name__}: {e}"
            JOBS[job_id]["finished_at"] = _now_iso()


# ---------- API ----------

class RunRequest(BaseModel):
    max_stocks: int | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        raise HTTPException(500, "static/index.html missing")
    return p.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "ts": _now_iso()}


@app.post("/api/run")
def post_run(req: RunRequest) -> dict:
    job_id = uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "max_stocks": req.max_stocks,
            "queued_at": _now_iso(),
        }
    Thread(target=_execute, args=(job_id, req.max_stocks), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


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
def latest() -> dict:
    if not LATEST:
        raise HTTPException(404, "no result yet")
    return LATEST


@app.get("/api/dashboard.png")
def dashboard_png():
    p = OUTPUT_DIR / "dashboard.png"
    if not p.exists():
        raise HTTPException(404, "dashboard not generated yet")
    return FileResponse(p, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
