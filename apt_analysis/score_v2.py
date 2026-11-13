"""v2 점수: 단지 자체의 가치 (예산 적합·을지로 거리 제외, 학군·교통·지형 추가).

현재 데이터가 있는 항목만 계산하고, 없는 항목(학군·교통·상권)은 빼서 남은 가중치로 재정규화한다.
  python -m apt_analysis.score_v2
입력: data/scorecard_all.csv, data/terrain.csv   산출: data/scorecard_v2_all.csv, data/scorecard_v2.csv(필터 통과)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import metrics_cfg as M

DATA = Path(__file__).resolve().parent.parent / "data"
# 항목별 현재 데이터 보유 여부 (새 데이터가 붙으면 여기에 추가)
V2_AVAILABLE = ["loc", "defense", "liquidity", "age", "terrain", "life"]
COL = {"loc": "s_loc", "defense": "s_defense", "liquidity": "s_liquidity", "age": "s_age",
       "terrain": "s_terrain", "life": "s_income"}      # 생활수준은 현재 소득만(비어 있으면 3점)


def terrain_score(pct: float | None) -> int:
    if pct is None or pd.isna(pct):
        return 3                                         # 좌표 없음 → 중립
    for hi, s in M.TERRAIN_PCT_BINS:
        if pct <= hi:
            return s
    return 1


def main() -> int:
    a = pd.read_csv(DATA / "scorecard_all.csv", encoding="utf-8-sig")
    t = pd.read_csv(DATA / "terrain.csv", encoding="utf-8-sig")[["cid", "elev_m", "slope500_pct"]]
    a = a.merge(t, on="cid", how="left")
    a["slope_pct_rank"] = a.slope500_pct.rank(pct=True, method="average")     # 낮을수록 평탄
    a["s_terrain"] = a.slope_pct_rank.map(terrain_score)
    w = {k: M.WEIGHTS_V2[k] for k in V2_AVAILABLE}
    tot = sum(w.values())
    a["score_v2"] = sum(a[COL[k]] * v for k, v in w.items()) / tot
    a["score_v2_100"] = (a.score_v2 / 5 * 100).round(1)
    a["score_v2"] = a.score_v2.round(3)
    a["v2_coverage"] = tot                                # 반영된 가중치 합 (100 이면 전 항목 반영)
    a = a.sort_values(["pass_all", "score_v2"], ascending=[False, False]).reset_index(drop=True)
    a.to_csv(DATA / "scorecard_v2_all.csv", index=False, encoding="utf-8-sig")
    p = a[a.pass_all].copy()
    p.insert(0, "rank_v2", range(1, len(p) + 1))
    p.to_csv(DATA / "scorecard_v2.csv", index=False, encoding="utf-8-sig")
    print(f"v2: 반영 가중치 {tot}/100 ({', '.join(V2_AVAILABLE)}) · 통과 {len(p)}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
