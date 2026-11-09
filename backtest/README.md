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
