# minervini

Minervini 스타일 모멘텀 / 브레이크아웃 leading indicator 모니터 — **KOSPI + NASDAQ** 양 시장, **개별 종목 돌파매매 분석**, **AI 챗 패널**, **Postgres 기반 분석 기록**을 한 화면에 통합한 FastAPI + 단일 HTML 도구.

## 무엇을 측정하나

### 시장 집계 (KOSPI / NASDAQ 탭)

| Indicator | 의미 |
|---|---|
| `% > SMA200`, `% > SMA50` | 시장 폭 (breadth) |
| `% Trend Template` 통과 비율 | Minervini 8-criteria 통과 종목 비중 |
| 신고가 - 신저가 | Net New Highs |
| Pivot / Quality 브레이크아웃 카운트 | 일별 시그널 발생량 |
| **20거래일 롤링 BSR** (브레이크아웃 success rate) | +10% 도달이 -8% 손절보다 먼저 일어난 비율 (경로 의존) |
| **MFHR** (모멘텀 상위 데실 hit rate) | RS rank 상위 10% 종목의 forward return 양호 비율 |
| **Sector rotation score** | 섹터별 BSR/MFHR의 사분위 spread 평균 — 섹터 선택의 효용 |
| **섹터 랭킹** | 최신일 섹터별 BSR/MFHR + composite_rank → 주도/약세 섹터 식별 |

BSR · MFHR · rotation_score 가 핵심 leading indicator. 시장이 모멘텀에 호의적인지 비호의적인지를 종목 수준 결과로 검증한다.

### 단일 종목 (종목 분석 탭)

티커 하나 입력 → 즉시 출력:
- 트렌드 템플릿 8개 조건 통과 여부 + 실제 값 비교
- 현재 셋업 플래그 (52w 신고가 근접, VCP 압축, Pivot 돌파, Quality 돌파, 거래량 급증 등)
- 매매 셋업 (Pivot 가격 / Stop −8% / Stop −2×ATR / Target +10%/+20%)
- 캔들스틱 + SMA50/150/200 + 52w high + 과거 돌파 시그널 마커
- 거래량
- 과거 돌파 히스토리 (각 시점의 +5d/+20d/+60d 수익률, 20d MDD, HIT/MISS 판정)

## 화면 구성

상단 4-way 탭:

| 탭 | 내용 |
|---|---|
| **KOSPI** | KOSPI 시장 집계 + 섹터 회전 + 오늘의 Picks |
| **NASDAQ** | 동일 분석을 NASDAQ 종목 universe로 |
| **종목 분석** | 단일 티커 돌파매매 분석 (KOSPI 6자리 / NASDAQ 심볼 자동 감지) |
| **기록** | DB에 저장된 모든 과거 분석 (시장 집계 + 단일 종목) 테이블. 행 클릭 시 그 시점 화면 그대로 복원 |

화면 우하단에 **AI 챗 패널** — 드래그로 이동, 코너로 리사이즈, localStorage에 위치·크기 보존. 모델 드롭다운에서 Anthropic 직접(Sonnet 4.6 / Opus 4.7 / Haiku 4.5) 또는 OpenRouter 경유 모델들(GPT-4o, Gemini 2 Flash, DeepSeek, Gemma 4 26B/31B, gpt-oss 120B, GLM 4.5 Air, MiniMax M2.5)을 선택. 현재 탭의 최신 결과를 컨텍스트로, 도구를 호출해 KPI / 시계열 / 섹터 랭킹 / picks / CSV를 직접 조회해 답한다 (응답 제약 없음).

## 빠른 시작

### CLI

```bash
pip install -r requirements.txt

# KOSPI 전체
python main.py

# 빠른 테스트
python main.py --max-stocks 50

# NASDAQ 분석
python main.py --market NASDAQ --max-stocks 200

# 과거 시점으로 백테스트하듯 실행
python main.py --as-of 2024-12-30 --max-stocks 100

# 출력 위치 지정
python main.py --out-dir ./report
```

### 웹 UI (FastAPI + 단일 HTML)

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
# 브라우저에서 http://127.0.0.1:8000 접속
```

웹 UI 동작:
1. 시장 탭 선택 → 종목 수 / 기준일 / 캐시 무시 토글 → **실행** → 백그라운드 잡 (2초 폴링)
2. KPI 카드 + Plotly 인터랙티브 차트 6종 (시장지수 / Breadth / 신고가-신저가 / 브레이크아웃 카운트 / 롤링 hit rate / **rotation_score**)
3. 섹터 랭킹 테이블 (BSR / MFHR / composite_rank)
4. 오늘의 Quality 브레이크아웃 종목 (Code · Name · **Sector** · Close · RS · 리턴 · Vol)
5. 서버 사이드 Matplotlib 대시보드 PNG
6. **종목 분석 탭** — 티커 한 줄로 검색해서 트렌드템플릿 스코어카드 + 매매 셋업 + 캔들 차트 + 돌파 히스토리
7. **기록 탭** — 과거 모든 분석을 한 테이블로 (시장/단일 혼합). 행 클릭 → 그 시점 화면 복원
8. **AI 챗 패널** — 우하단, 어디서든 사용 가능

API 엔드포인트:

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/` | 단일 HTML 프론트엔드 |
| `GET` | `/api/health` | 상태 + DB 활성 여부 |
| `POST` | `/api/run` | 시장 집계 파이프라인 실행 (`{market, max_stocks, as_of, force_refresh}`) |
| `GET` | `/api/status/{job_id}` | 잡 상태 (queued / running / completed / failed) + source(cache/fresh) |
| `GET` | `/api/result/{job_id}` | 완료된 잡의 summary |
| `GET` | `/api/latest?market=KOSPI` | 시장별 최근 결과 |
| `GET` | `/api/dashboard.png?market=KOSPI` | 시장별 Matplotlib PNG |
| `POST` | `/api/single` | 단일 티커 분석 (`{ticker, market?, as_of?}`) |
| `GET` | `/api/history?market=...&limit=...` | 시장 집계 + 단일 종목 합쳐서 최신순 |
| `GET` | `/api/history/{id}` | 시장 집계 분석 상세 |
| `GET` | `/api/history/{id}/dashboard.png` | 저장된 시점의 PNG (DB BYTEA) |
| `GET` | `/api/history/single/{id}` | 단일 종목 분석 상세 |
| `POST` | `/api/chat` | 챗 (`{messages, model, provider, market}`) |
| `GET` | `/docs` | Swagger UI |

출력 (`output_<market>/`):
- `dashboard.png` — 5단 시각화
- `leading_indicators.csv` — 일별 패널
- `breakout_signals.csv` — 시그널 raw + forward returns
- `breakout_rolling_hit_rate.csv` — BSR
- `factor_decile_hit_rate.csv` — MFHR
- `sector_breakout_hit_rate.csv` / `sector_factor_hit_rate.csv` — 섹터별 BSR/MFHR
- `rotation_score.csv` — 섹터 dispersion 시계열
- `sector_ranking.csv` — 최신일 섹터 랭킹
- `today_picks.csv` — 오늘 RS≥70 Quality 브레이크아웃 종목

## 환경 변수

| 변수 | 용도 | 미설정 시 |
|---|---|---|
| `DATABASE_URL` | Railway Postgres 연결 → 분석 기록 캐싱 + 히스토리 탭 | 캐시 off, 매번 새로 계산 |
| `ANTHROPIC_API_KEY` | 챗 패널의 Claude 직접 호출 (Sonnet/Opus/Haiku) | 해당 모델 사용 불가 |
| `OPENROUTER_API_KEY` | 챗 패널의 OpenRouter 모델들 | 해당 모델 사용 불가 |

## Railway Postgres 캐싱

같은 (`market`, `as_of`, `max_stocks`, `config_hash`) 입력으로 들어온 실행은 DB에서 즉시 조회 — 재계산하지 않는다. Config 분석 파라미터를 만지면 hash가 바뀌어 자동 무효화.

설정:
1. Railway 프로젝트에서 **+ New → Database → PostgreSQL** 추가.
2. `DATABASE_URL`이 자동으로 같은 프로젝트의 minervini 서비스에 주입됨.
3. 첫 배포 시 `db.init()`이 `pipeline_runs` + `single_runs` 테이블 자동 생성 (idempotent `IF NOT EXISTS`).
4. `/api/health` 호출 → `{"db": {"enabled": true}}` 확인.

저장 항목 (v1 lean):
- `pipeline_runs`: market / as_of / max_stocks / config_hash / config_json / summary JSONB / dashboard_png BYTEA / **client_ip** / 타이밍
- `single_runs`: ticker / market / as_of / result JSONB / client_ip / 타이밍

캐시 무시: UI 컨트롤바의 **"캐시 무시"** 체크박스 → `force_refresh=true` 로 전송 → INSERT/UPDATE 로 덮어쓰기.

## 모듈

| 파일 | 역할 |
|---|---|
| `config.py` | `Config` dataclass + `for_market()` factory (KOSPI 2000원 / NASDAQ $5, 캐시 디렉토리 자동) |
| `data.py` | 시장 추상화: `get_listing(cfg)`, `get_index(cfg, ...)`, `get_sector_map(cfg)`. OHLCV 캐시·병렬 다운로드 |
| `factors.py` | SMA, 1/3/6/12개월·12-1 모멘텀, IBD-style RS, Minervini 트렌드 템플릿, 횡단면 퍼센타일 랭크 |
| `breakout.py` | Pivot / 52주 신고가 / 거래량 급증 / VCP 압축 / `quality_breakout` |
| `hit_rate.py` | Forward return + 경로 의존적 hit 라벨, 시그널·팩터 데실 hit rate, **섹터별 BSR/MFHR**, **rotation_score**, **섹터 랭킹** |
| `monitor.py` | 오케스트레이션 + leading indicator 패널 빌드 (시장 무관) |
| `single_analysis.py` | 단일 티커 돌파매매 분석 (트렌드 템플릿 스코어카드 + 매매 셋업 + 돌파 히스토리) |
| `plot.py` | 5단 대시보드 PNG (시장명 자동 반영) |
| `main.py` | CLI 진입점 (`--market`, `--max-stocks`, `--as-of`, `--out-dir`) |
| `app.py` | FastAPI 백엔드 (per-market 잡/캐시, 챗, 히스토리, 단일 분석) |
| `db.py` | psycopg3 ConnectionPool, 스키마 부트스트랩, `pipeline_runs` + `single_runs` lookup/save |
| `chat_agent.py` | Anthropic / OpenRouter 양방향 챗 에이전트 + read-only 도구 7종 (대시보드 데이터 조회) |
| `static/index.html` | 단일 파일 프론트엔드 — 4 탭, 6 차트, 챗 패널 (드래그/리사이즈) |
| `test_smoke.py` | 합성 데이터로 핵심 로직 검증 (8단계, 네트워크 불필요) |

## 환경

- Python 3.11 ~ 3.14
- `finance-datareader`(주) 또는 `pykrx`(폴백) 중 하나 이상 필요
- Postgres 캐싱은 선택 — `DATABASE_URL` 없으면 graceful no-op
- 챗은 선택 — API 키 없으면 해당 provider 비활성
- Windows 콘솔이 cp949이면 stdout 한글이 깨져 보일 수 있다 (`chcp 65001` + `set PYTHONIOENCODING=utf-8`로 해결). CSV/PNG/웹 UI 출력은 영향 없음.

## 라이선스

MIT
