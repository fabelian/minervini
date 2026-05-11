"""대시보드 우하단 챗 패널 전용 에이전트.

agenticCRMFlow의 ChatAgent / BaseAgent 패턴을 minervini KOSPI 대시보드에 맞게
이식한 단일 모듈. 두 provider를 지원한다:
  - "anthropic"  : Anthropic SDK (tool_use 패턴)
  - "openrouter" : OpenRouter / OpenAI 호환 (function calling 패턴)

도구는 모두 read-only: 최신 대시보드 요약, picks, 섹터 랭킹, 시계열, 산출 CSV.
시스템 프롬프트는 토픽/거부 제약을 두지 않고 도구 사용만 가이드한다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output_kospi"

SYSTEM_PROMPT = """당신은 KOSPI 모멘텀/브레이크아웃 모니터 대시보드에 통합된 챗 어시스턴트입니다.

다음 도구로 현재 대시보드 데이터를 읽을 수 있습니다:
- get_latest_summary: 최신 KPI · as-of 날짜 · 집계 종목 수 등 요약 dict
- get_picks: 오늘의 Quality 브레이크아웃 종목 (code/name/sector/RS/리턴 등)
- get_sector_ranking: 최신일 섹터별 BSR/MFHR + composite_rank
- list_series: 보유 시계열 키 목록
- get_series: 특정 시계열을 (date, value) 포인트 배열로
- list_csv / read_csv: output_kospi 디렉토리 CSV 직접 조회

사용자 질문이 대시보드 데이터를 필요로 하면 도구를 호출해 실제 값으로 답하세요.
일반 질문(시장 개념, 코딩, 기타 어떤 주제든)도 자유롭게 답하세요.
응답 형식·주제·거부 응답에 대한 별도 제약은 없습니다. 한국어가 기본이지만 사용자 언어에 맞추세요."""


# ---------- Dashboard context (도구가 참조하는 런타임 상태) ----------

class DashboardContext:
    def __init__(self, latest_summary: dict | None = None,
                 output_dir: Path | None = None):
        self.latest_summary = latest_summary or {}
        self.output_dir = output_dir or OUTPUT_DIR

    def get_latest_summary(self) -> dict:
        return self.latest_summary

    def get_picks(self) -> list:
        return self.latest_summary.get("picks", []) or []

    def get_sector_ranking(self) -> list:
        return self.latest_summary.get("sector_ranking", []) or []

    def list_series(self) -> dict:
        return {"keys": list((self.latest_summary.get("series") or {}).keys())}

    def get_series(self, key: str, n_last: int = 60) -> dict:
        s = (self.latest_summary.get("series") or {}).get(key) or []
        if n_last and n_last > 0:
            s = s[-n_last:]
        return {"key": key, "n": len(s), "points": s}

    def list_csv(self) -> dict:
        if not self.output_dir.exists():
            return {"files": []}
        return {"files": sorted(p.name for p in self.output_dir.glob("*.csv"))}

    def read_csv(self, name: str, head: int = 50) -> dict:
        path = self.output_dir / name
        if not path.exists():
            return {"error": f"{name} 파일이 없습니다."}
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        head = max(1, min(int(head or 50), 500))
        return {
            "file": name,
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
            "head": df.head(head).to_dict(orient="records"),
        }


TOOL_SCHEMAS = [
    {"name": "get_latest_summary",
     "description": "최신 대시보드 요약(KPI 값, as-of 날짜, 등) JSON을 반환.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_picks",
     "description": "오늘의 Quality 브레이크아웃 종목(code/name/sector/RS/리턴 등) 리스트.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_sector_ranking",
     "description": "최신일 섹터별 BSR/MFHR 및 composite_rank 랭킹 테이블.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_series",
     "description": "보유한 시계열 키 목록(예: pct_above_sma200, bo_hit_rate, rotation_score, bsr_spread, mfhr_spread, kospi).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_series",
     "description": "특정 시계열을 (date, value) 포인트 배열로 반환.",
     "input_schema": {
         "type": "object",
         "properties": {
             "key": {"type": "string", "description": "list_series가 반환한 키 중 하나"},
             "n_last": {"type": "integer", "description": "최근 N개만. 기본 60. 0이면 전체."},
         },
         "required": ["key"],
     }},
    {"name": "list_csv",
     "description": "output_kospi 디렉토리의 CSV 파일 목록을 반환.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "read_csv",
     "description": "output_kospi 디렉토리의 CSV를 읽어 컬럼 + 상위 N행(JSON records)을 반환.",
     "input_schema": {
         "type": "object",
         "properties": {
             "name": {"type": "string", "description": "예: sector_ranking.csv"},
             "head": {"type": "integer", "description": "반환 행 수. 기본 50, 최대 500."},
         },
         "required": ["name"],
     }},
]


def _dispatch_tool(ctx: DashboardContext, name: str, args: dict) -> Any:
    try:
        if name == "get_latest_summary":
            return ctx.get_latest_summary()
        if name == "get_picks":
            return ctx.get_picks()
        if name == "get_sector_ranking":
            return ctx.get_sector_ranking()
        if name == "list_series":
            return ctx.list_series()
        if name == "get_series":
            return ctx.get_series(args["key"], int(args.get("n_last", 60)))
        if name == "list_csv":
            return ctx.list_csv()
        if name == "read_csv":
            return ctx.read_csv(args["name"], int(args.get("head", 50)))
        return {"error": f"unknown tool: {name}"}
    except KeyError as e:
        return {"error": f"필수 인자 누락: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------- ChatAgent ----------

class ChatAgent:
    def __init__(self, model: str | None = None, provider: str = "anthropic",
                 context: DashboardContext | None = None):
        self.provider = (provider or "anthropic").lower()
        self.context = context or DashboardContext()
        self.tools = TOOL_SCHEMAS
        if self.provider == "anthropic":
            from anthropic import Anthropic
            self.model = model or "claude-sonnet-4-6"
            self.client = Anthropic()
        elif self.provider == "openrouter":
            from openai import OpenAI
            api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY 환경변수가 필요합니다.")
            self.model = model or "openai/gpt-4o"
            self.client = OpenAI(api_key=api_key,
                                 base_url="https://openrouter.ai/api/v1")
        else:
            raise ValueError(f"unknown provider: {self.provider}")

    def chat(self, messages: list[dict], max_iter: int = 10) -> str:
        safe: list[dict] = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                safe.append({"role": role, "content": content})
        if not safe:
            return "질문을 입력해 주세요."
        if self.provider == "anthropic":
            return self._chat_anthropic(safe, max_iter)
        return self._chat_openrouter(safe, max_iter)

    # ----- Anthropic -----
    def _chat_anthropic(self, init: list[dict], max_iter: int) -> str:
        messages = list(init)
        out: list[str] = []
        iters = 0
        while True:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=messages,
            )
            for b in resp.content:
                if b.type == "text" and b.text and b.text.strip():
                    out.append(b.text)
            if resp.stop_reason == "end_turn":
                return "\n".join(out).strip()
            if resp.stop_reason == "tool_use":
                iters += 1
                if iters > max_iter:
                    return ("\n".join(out).strip() + "\n\n(도구 반복 상한 초과)").strip()
                tool_results = []
                for b in resp.content:
                    if b.type != "tool_use":
                        continue
                    r = _dispatch_tool(self.context, b.name, b.input or {})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": json.dumps(r, ensure_ascii=False, default=str),
                    })
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})
                continue
            # max_tokens / 그 외
            return "\n".join(out).strip()

    # ----- OpenRouter -----
    def _chat_openrouter(self, init: list[dict], max_iter: int) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(init)
        out: list[str] = []
        openai_tools = [
            {"type": "function",
             "function": {
                 "name": t["name"],
                 "description": t.get("description", ""),
                 "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
             }}
            for t in self.tools
        ]
        iters = 0
        while True:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=openai_tools,
            )
            if not getattr(resp, "choices", None):
                err = getattr(resp, "error", None) or "no choices"
                return f"OpenRouter 응답 오류: {err}"
            choice = resp.choices[0]
            msg = choice.message
            fr = choice.finish_reason
            if msg.content:
                out.append(msg.content)
            if fr == "stop":
                return "\n".join(out).strip()
            if fr == "tool_calls" and msg.tool_calls:
                iters += 1
                if iters > max_iter:
                    return ("\n".join(out).strip() + "\n\n(도구 반복 상한 초과)").strip()
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    r = _dispatch_tool(self.context, tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(r, ensure_ascii=False, default=str),
                    })
                continue
            return "\n".join(out).strip()
