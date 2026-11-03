# 서울 아파트 투자가치 분석 프로젝트

## 목표
투자가치 최우선(입지 좋고 많이 오를 곳), 실거주 겸용.
- 예산 10억 (현금 5억 + 대출 5억), 생애최초, 연봉 1억
- 을지로 출퇴근 (비중 낮음)

## 데이터
- `github.com/woongcomi/house_pjt` 클론 사용
- `web/public/data/gu/*.json`: 국토부 실거래가 기반 단지×월 지수 (CSR 형식)
  - `ids`, `off`, `dt`(월 델타, base부터), `p`(만원/평, 전용), `n`(3개월 풀링 거래수), `amt`(대표 평형 금액, 만원)
  - 서울 25개 구 + 분당(41135) 포함

## 회복률
- 2022.12 → 2025.9 수익률 (±2개월 내 가장 가까운 월, 거래수 가중)
- 고점 = 2021.1 ~ 2022.6 최대값
- 고점 대비 = 2025.9 / 고점 − 1

## 확장 스코어카드 필터
- 회복률 0% 이상
- 세대수 200 이상
- 최근 12개월 월평균 거래 ≥ 세대수의 0.2% (기존 0.5%에서 사용자 변경)
- 대표 평형 전용 40㎡ 이상
- 도시형생활주택 제외

## 점수 (1~5점, 가중치 %)
| 항목 | 가중치 | 기준 |
|---|---|---|
| 입지 | 25 | 동 평당가 백분위 |
| 하락장 방어 | 20 | 2025.9 고점 대비: 0%↑ 5, −5%↑ 4, −10%↑ 3, −15%↑ 2 (그 외 1) |
| 생활수준 | 15 | 구 1인당 종합소득 |
| 유동성 | 15 | 2022.7~2023.6 월평균 거래 |
| 연식 | 10 | |
| 예산 적합 | 10 | 84㎡ 환산 금액 |
| 을지로 직선거리 | 5 | |

## 작업 규칙
- 스코어카드 스크립트 결과는 `data/`에 저장
- 세대수: 공공데이터포털 K-apt 단지 기본정보 API (`AptBasisInfoServiceV4`, `kaptdaCnt`). API 키는 환경변수 `DATA_GO_KR_KEY` (코드/커밋에 키 금지)
- 동탄(화성시, 41590): 국토부 아파트 매매 실거래가 API로 수집 → 같은 방식으로 지수화 후 추가

## 실행 방법 (apt_analysis/)
```bash
git clone https://github.com/woongcomi/house_pjt <경로>      # 저장소 밖(예: /tmp)에 클론
export DATA_GO_KR_KEY=...                                    # 커밋 금지
python -m apt_analysis.dongtan --house-pjt <경로>            # 동탄 수집·지수화 → data/dongtan/gu/
python -m apt_analysis.kapt --data-dir <경로>/web/public/data/gu   # 세대수 → data/households.csv
python -m apt_analysis.scorecard --data-dir <경로>/web/public/data/gu
python -m unittest tests_apt.test_apt
```
- 결과: `data/scorecard.csv`(필터 통과), `data/scorecard_all.csv`(전체). `households.csv`가 없으면 `scorecard_preliminary.csv`(세대수·회전율 필터 미적용)
- `data/inputs/gu_income.csv`의 `income_per_capita`(구 1인당 종합소득)는 사용자가 채워야 함. 비어 있으면 생활수준 3점(중립) 처리
- 사용자가 정하지 않아 가정한 구간표(백분위·연식·예산·거리 등)는 `apt_analysis/metrics_cfg.py`에 모아둠
- 월평균 거래는 `n`(3개월 풀링)/3 으로 추정, "최근 12개월"은 2025.9 종료 기준
