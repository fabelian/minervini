# minervini

Simple Minervini-type KOSPI leadership health monitor.

성장/모멘텀 팩터의 hit rate와 브레이크아웃 success rate를 leading indicator로 추적하는 KOSPI 한정 파이썬 도구.

## 무엇을 측정하나

| Indicator | 의미 |
|---|---|
| `% > SMA200`, `% > SMA50` | 시장 폭 (breadth) |
| `% Trend Template` 통과 비율 | Minervini 8-criteria 통과 종목 비중 |
| 신고가 - 신저가 | Net New Highs |
| Pivot / Quality 브레이크아웃 카운트 | 일별 시그널 발생량 |
| **20거래일 롤링 브레이크아웃 hit rate** | +10% 도달이 -8% 손절보다 먼저 일어난 비율 (경로 의존) |
| **모멘텀 상위 데실 hit rate** | RS rank 상위 10% 종목의 forward return 양호 비율 |

마지막 두 항목이 핵심 leading indicator. 시장이 모멘텀에 호의적인지 비호의적인지를 종목 수준 결과로 검증한다.

## 빠른 시작

### CLI

```bash
pip install -r requirements.txt
python main.py                     # 전체 KOSPI (~10-15분, 캐시 후 ~1분)
python main.py --max-stocks 50     # 빠른 시험
python main.py --out-dir ./report  # 출력 위치 지정
```

### 웹 UI (FastAPI + 단일 HTML)

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
# 브라우저에서 http://127.0.0.1:8000 접속
```

웹 UI 동작:
1. 종목 수를 지정하고 **실행** 버튼 → 백그라운드 잡 시작 (2초 폴링)
2. KPI 카드 + Plotly 인터랙티브 차트 5종 (KOSPI / Breadth / 신고가-신저가 / 브레이크아웃 카운트 / 롤링 hit rate)
3. 오늘의 Quality 브레이크아웃 종목 테이블
4. 서버 사이드 Matplotlib 대시보드 PNG 임베드

API 엔드포인트:

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/` | 단일 HTML 프론트엔드 |
| `POST` | `/api/run` | 파이프라인 실행 (`{"max_stocks": N}`), `{"job_id": ...}` 반환 |
| `GET` | `/api/status/{job_id}` | 잡 상태 (queued / running / completed / failed) |
| `GET` | `/api/result/{job_id}` | 완료된 잡의 요약 + 시계열 + picks |
| `GET` | `/api/latest` | 가장 최근 완료된 잡 결과 |
| `GET` | `/api/dashboard.png` | Matplotlib 대시보드 PNG |
| `GET` | `/docs` | Swagger UI |

출력:
- `output_kospi/dashboard.png` — 5단 시각화
- `output_kospi/leading_indicators.csv` — 일별 패널
- `output_kospi/breakout_signals.csv` — 시그널 raw + forward returns
- `output_kospi/breakout_rolling_hit_rate.csv` — 핵심 leading indicator
- `output_kospi/factor_decile_hit_rate.csv` — 모멘텀 데실 hit rate
- `output_kospi/today_picks.csv` — 오늘 RS≥70 Quality 브레이크아웃 종목

## 모듈

| 파일 | 역할 |
|---|---|
| `config.py` | 파라미터 (이동평균, 브레이크아웃 임계값, +10%/-8% 룰, 롤링 윈도우) |
| `data.py` | KOSPI 종목·OHLCV 로더 (FinanceDataReader 우선, pykrx 폴백, 캐시·병렬 다운로드) |
| `factors.py` | SMA, 1/3/6/12개월·12-1 모멘텀, IBD-style RS 컴포지트, Minervini 트렌드 템플릿, 횡단면 퍼센타일 랭크 |
| `breakout.py` | Pivot / 52주 신고가 / 거래량 급증 / VCP 압축 / `quality_breakout` |
| `hit_rate.py` | Forward return + 경로 의존적 hit 라벨, 시그널·팩터 데실 hit rate 집계 |
| `monitor.py` | 오케스트레이션 + leading indicator 패널 빌드 |
| `plot.py` | 5단 대시보드 PNG |
| `main.py` | CLI 진입점 |
| `app.py` | FastAPI 백엔드 (백그라운드 잡 + JSON API + 정적 HTML) |
| `static/index.html` | 단일 파일 프론트엔드 (Plotly via CDN, 빌드 불필요) |
| `test_smoke.py` | 네트워크 없이 합성 데이터로 핵심 로직 검증 |

## 환경

- Python 3.11 ~ 3.14
- `finance-datareader`(주) 또는 `pykrx`(폴백) 중 하나 이상 필요
- Windows 콘솔이 cp949이면 stdout 한글이 깨져 보일 수 있다 (`chcp 65001` + `set PYTHONIOENCODING=utf-8`로 해결). CSV/PNG 출력은 영향 없음.

## 라이선스

MIT
