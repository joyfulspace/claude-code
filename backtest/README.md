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

## 결과 설명용 그림 (한글)
    pip install koreanize-matplotlib
    python3 backtest/run_sell_lab.py && python3 backtest/explain_charts.py
`results/explain_1~5_*.png`: 규칙 예시, 규칙 비교, 구간별 회피, 지표 구분력, 현재 위치.

## 지표 하나씩 확인 조건으로 평가
    python3 backtest/run_single_indicator.py --thr 70
(고점 대비 -8% 이하) AND (지표 하나 상위 30%) 규칙 17개를 기본 규칙과 비교. `results/single_1_지표별.png`, `single_2_구간별.png`.

## 지표별 파라미터 최적화 후 비교
    python3 backtest/run_param_opt.py && python3 backtest/explain_param_opt.py
점수 = 회피비율과 유효율의 조화평균. 지표 창·임계 백분위·방향·하락 D를 격자 탐색(주변 평균이 높은 안정 구간 선택), 기간 교차검증과
가짜 지표 최적화(우연 기준선)로 검증. `results/opt_1~4_*.png`, `param_opt_summary.csv`.

## '하락을 잘 피했다' 점수 정의 (avoid_score.py)
지표 단독 신호 → 다음날 시가 매도(재진입 없음), 120거래일 측정 창, 신호 간 120일 간격.
점수 = 큰 하락 구간별 평균 피한 손실 - λ × 모든 신호의 평균 놓친 상승 (금액형/비율형, λ=0.25/0.5/1). 그림: `python3 backtest/explain_avoid_def.py`.

## 지표별 최적화 + 검증 (확정 정의)
    python3 backtest/run_avoid_opt.py && python3 backtest/explain_avoid_opt.py
창 길이 × 임계(상위 5~30%) 탐색, 기간 교차·구간별 교차·가짜 지표 기준선 검증, 신호 수가 같은 무작위 신호와 비교. `results/avo_1~3_*.png`.

## 신호 발생 시점 시각화
    python3 backtest/explain_signal_timeline.py
λ=0.1 최적 파라미터의 지표별 신호 시점을 큰 하락 구간과 함께 표시. `results/sig_1_전체시계열.png`, `sig_2_최근시계열.png`.

## 지표 조합 평가 ('n개 이상 울리면 진짜 신호')
    python3 backtest/run_combo.py && python3 backtest/explain_combo.py
최근 W거래일 안에 신호가 울린 지표 수 ≥ n (n=1~16, W=5/20/60), 울린 그룹 수 변형, 파라미터 기본값/전체최적/교차검증 3가지, 조합 count 를 무작위 이동한 우연 기준선과 비교. `results/combo_1~6_*.png`.

## 정정: 우연 기준선 계산 (run_combo.py, run_avoid_opt.py)
신호를 시간축 전체로 돌려 밀면 평가 시작일(2006) 이전으로 밀린 신호가 사라져 가짜 신호가 적어지고 기준선이 낮아지던 오류가 있었음.
`roll_win`으로 평가 구간 안에서만 원형 이동하도록 수정(가짜 신호 수가 진짜와 같아짐). `python3 backtest/explain_group_null.py` 로 원리 그림 생성.

## 원본(raw) 시계열 시각화
    python3 backtest/explain_raw_series.py
예탁금·신용융자·신용융자/예탁금·반대매매 비중·원/달러·S&P500·미국 10년 금리·VIX·코스닥/코스피를 KOSPI와 함께 표시. `results/raw_1~3_*.png`.
주의: 반대매매 비중은 2006-04 이전 ≈0, 2023-10-19 69% 급등, 2023-12 에 수준이 ~9%→~0.6% 로 이동(집계 기준 변경 의심).
