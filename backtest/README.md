# backtest — 국내 주식 규칙 백테스트

## 구조
- `data.py` 일봉 로더 (CSV 캐시 → 공공데이터포털 API). `--code SYNTH` 는 합성 데이터(점검용)
- `engine.py` 롱온리, 신호 다음날 시가 체결, 수수료/거래세/슬리피지 반영
- `metrics.py` 누적수익률, CAGR, 변동성, 샤프, MDD, 거래횟수, Buy&Hold 비교
- `strategies/` 규칙 1개 = 파일 1개. `signal(df, **params)` 가 1(보유)/0(현금) 시리즈를 반환
- `run.py` 실행기

## 실행
    python3 backtest/run.py --strategy ma_cross --code 005930 --start 2020-01-01 --end 2025-12-31 --split 2023-01-01
    python3 backtest/run.py --strategy ma_cross --code SYNTH --param fast=10 --param slow=40

## 새 규칙 추가
`strategies/내규칙.py` 에 `PARAMS` 와 `signal(df, ...)` 를 작성. df 컬럼: open/high/low/close/volume.
신호는 당일 종가까지의 정보만 써야 하며, 체결은 엔진이 다음날 시가로 처리한다.

## 주의
- 공공데이터 `clpr` 는 수정주가가 아니라 액면분할/배당 구간에 왜곡이 있을 수 있음
- 거래세율은 시기별로 다르니 `engine.Costs` 에서 확인·조정
- 데이터 키: 환경변수 `DATA_GO_KR_KEY`

## 포트폴리오(다종목) 백테스트
    python3 backtest/run_portfolio.py --synth --split 2018-01-01
    python3 backtest/run_portfolio.py --codes 005930,000660,... --split 2022-01-01 --param thresh=-0.05
필요 데이터(`data/`): `KOSPI.csv`, `SPX.csv` (date, close), `valuation_pbr.csv`, `valuation_per.csv`
(행=날짜, 열=종목코드; 공시 이후 시점만), 종목별 `{코드}.csv`.
전략 `strategies/crash_value.py`: 한·미 동시 급락(기본 1개월 -5%) 시 보유 종목 중 고평가 매도 → 저평가 추가 매수 (숏 없음).
PBR / PER / 조합(순위 평균) 결과를 기준(균등 보유)·KOSPI와 한 표로 비교.

## 지수 데이터 (야후 파이낸스)
    python3 backtest/fetch_indices.py 2000-01-01
KOSPI(^KS11)와 S&P500(^GSPC)를 `data/KOSPI.csv`, `data/SPX.csv`로 저장. 네트워크 정책에서 `query1.finance.yahoo.com` 허용 필요.
야후는 비공식 API라 구조가 바뀌거나 막힐 수 있음.

## KOSPI 전액 매도 알림 (위험 점수)
    python3 backtest/run_risk_exit.py --hi 85 --lo 60 --days 3
시장 국면·심리(VIX, RSI)·거래량 지표를 과거 대비 백분위로 점수화(`risk_exit.py`), 점수가 높으면 전액 매도 / 점진 감축을 Buy&Hold와 비교.
결과: `results/risk_exit.png`(알림 표시 차트), `results/risk_exit_alerts.csv`, `results/risk_exit_signals.csv`. 수급 지표는 미반영.

## 추세 이탈 × 수급 조합 비교
    python3 backtest/run_trend_supply.py
T1(200일선 이탈), T2(데드크로스), S(수급 단독), C1(이탈 AND 수급), C2(수급에 따라 기준선 변경)를 Buy&Hold와 비교. `results/trend_supply.png`.

## 큰 하락 조기경보 평가 (수익률이 아닌 하락 포착 중심)
    python3 backtest/run_crash_warning.py
라벨: 60거래일 내 -15% 하락. 지표별 AUC(블록 부트스트랩 CI, 기간별 방향 일관성), 하락 구간별 경보 시점, 워크포워드 로지스틱 검증. `results/crash_warning.png`.

## 하락 시작 후 대응 규칙 평가
    python3 backtest/run_decline_response.py
고점 대비 낙폭(DD), 200일선 이탈, 데드크로스, VIX/변동성, 한미 동반 하락 등 발동 규칙을 이벤트 단위(유효율·오경보·120일 후 수익)와 큰 하락 구간별 회피비율로 비교. `results/decline_response.png`.

## 매도 신호 실험실 (수급·시장민감도·심리)
    python3 backtest/fetch_indices.py   # KOSPI, S&P500, VIX, USD/KRW, 미국 10년 금리, 코스닥
    python3 backtest/run_sell_lab.py
하락 상태(고점 대비 -8% 이하)에서 17개 지표의 추가 하락 구분력(AUC)과, 그룹 확인 조건을 붙인 매도 규칙을 기본 규칙과 비교. `results/sell_lab.png`.
