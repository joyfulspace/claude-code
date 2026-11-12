"""확장 스코어카드.

  python -m apt_analysis.scorecard --data-dir <house_pjt>/web/public/data/gu
  (동탄: data/dongtan/gu 가 있으면 자동 포함)

입력(있으면 사용)
  data/households.csv        kapt.py 산출 (세대수, 유형)  → 없으면 '예비' 결과만 저장
  data/inputs/gu_income.csv  sgg,name,income_per_capita  → 없으면 소득 점수 중립(3) + 플래그

출력
  data/scorecard_all.csv  모든 단지: 지표·필터 플래그·점수
  data/scorecard.csv      필터 통과 단지, 총점 내림차순  (households.csv 없으면 scorecard_preliminary.csv)
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from . import metrics_cfg as M
from .common import (Complex, PYEONG_PER_M2, anchor, haversine_km, idx_to_ym,
                     load_all, monthly_trades, peak)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
URBAN_RE = re.compile(M.URBAN_NAME_RE)

# 동탄 신도시 법정동 (화성시 41590 중). 가정값 — 필요 시 수정.
DONGTAN_DONGS = {"반송동", "석우동", "능동", "청계동", "영천동", "중동", "신동", "목동",
                 "여울동", "산척동", "오산동", "송동", "장지동", "방교동", "금곡동", "기산동"}
DONGTAN_SGG = "41597"   # 2026 분리 후 동탄구 (구 화성시 41590)
DONGTAN_FALLBACK = (37.2000, 127.0950)     # 좌표 없는 동탄 단지용 대략 중심 (거리 점수는 어차피 1점대)


UNIVERSE_EXTRA = {"41135", DONGTAN_SGG}      # 분당 + 동탄. 서울(11*)은 항상 포함.


def in_universe(sgg: str) -> bool:
    """분석 대상: 서울 25개 구 + 분당 + 동탄 (house_pjt 에 같이 든 수지·영통 등은 제외)."""
    return sgg.startswith("11") or sgg in UNIVERSE_EXTRA


def bin_score(value: float, bins: list[tuple[float, int]]) -> int:
    for lo, s in bins:
        if value >= lo:
            return s
    return 1


def le_score(value: float, bins: list[tuple[float, int]]) -> int:
    for hi, s in bins:
        if value <= hi:
            return s
    return 1


def percentiles(values: dict[str, float]) -> dict[str, float]:
    """평균 순위 백분위 0~1 (동률은 평균 순위)."""
    keys = sorted(values, key=values.get)
    n = len(keys)
    if n <= 1:
        return {k: 0.5 for k in keys}
    out, i = {}, 0
    while i < n:
        j = i
        while j + 1 < n and values[keys[j + 1]] == values[keys[i]]:
            j += 1
        r = (i + j) / 2 / (n - 1)
        for k in keys[i:j + 1]:
            out[k] = r
        i = j + 1
    return out


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _price84_filled(c: Complex, p1: float | None) -> dict:
    """열람용 84㎡ 환산가. 2025.9(±2개월) 가격이 없으면 그 이전 최대 12개월 안의 가장 최근 월 가격을 이월, 그것도 없으면 이후 가장 가까운 월 가격을 사용(추정).
    점수·필터에는 쓰지 않는다(price84 는 그대로)."""
    if p1:
        return {"price84_filled": p1 * 84 * PYEONG_PER_M2, "price84_src": "2025-09"}
    prior = [(t, p) for t, p in zip(c.t, c.p) if M.T1 - 12 <= t < M.T1 - 2 and p]
    if prior:
        t, p = max(prior)
        return {"price84_filled": p * 84 * PYEONG_PER_M2, "price84_src": f"이월 {idx_to_ym(t)}"}
    later = [(t, p) for t, p in zip(c.t, c.p) if t > M.T1 + 2 and p]   # 이전이 없으면 이후 가장 가까운 월
    if later:
        t, p = min(later)
        return {"price84_filled": p * 84 * PYEONG_PER_M2, "price84_src": f"이후 {idx_to_ym(t)}"}
    return {"price84_filled": None, "price84_src": ""}


def compute(complexes: list[Complex], households: dict[str, dict],
            income: dict[str, float]) -> list[dict]:
    rows = []
    for c in complexes:
        if c.sgg == DONGTAN_SGG and c.dong not in DONGTAN_DONGS:
            continue
        p0, p1 = anchor(c, M.T0), anchor(c, M.T1)
        pk = peak(c, M.PEAK_LO, M.PEAK_HI)
        rec = (p1 / p0 - 1) if p0 and p1 else None
        dd = (p1 / pk - 1) if pk and p1 else None
        lat, lng = c.lat, c.lng
        coord_est = False
        if not lat or not lng:
            if c.sgg == DONGTAN_SGG:
                lat, lng = DONGTAN_FALLBACK
                coord_est = True
        dist = haversine_km(lat, lng, *M.EULJIRO) if lat and lng else None
        rows.append({
            "cid": c.cid, "sgg": c.sgg, "gu": c.gu_name, "dong": c.dong, "name": c.name,
            "build_year": c.build_year, "rep_area_m2": c.rep_area,
            "p_2022_12": p0, "p_2025_09": p1, "peak": pk,
            "recovery": rec, "from_peak": dd,
            "vol_12m": monthly_trades(c, M.T1 - M.RECENT_MONTHS + 1, M.T1),
            "vol_bear": monthly_trades(c, M.BEAR_LO, M.BEAR_HI),
            "price84": p1 * 84 * PYEONG_PER_M2 if p1 else None,
            **_price84_filled(c, p1),
            "dist_km": dist, "coord_estimated": coord_est,
            "households": None, "kapt_type": "", "kapt_matched": False,
        })
        h = households.get(c.cid)
        if h:
            r = rows[-1]
            r["kapt_matched"] = True
            r["households"] = int(float(h["households"])) if h.get("households") else None
            r["kapt_type"] = h.get("apt_type", "")
    return rows


def add_dong_percentile(rows: list[dict]) -> None:
    acc: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        if r["p_2025_09"]:
            acc[f"{r['sgg']}|{r['dong']}"].append((r["p_2025_09"], max(r["vol_bear"], 0.1)))
    dong_ppa = {k: sum(p * w for p, w in v) / sum(w for _p, w in v) for k, v in acc.items()}
    pct = percentiles(dong_ppa)
    for r in rows:
        k = f"{r['sgg']}|{r['dong']}"
        r["dong_ppa"] = dong_ppa.get(k)
        r["dong_pct"] = pct.get(k)


def filters(r: dict, have_households: bool) -> dict:
    f = {
        "f_recovery": r["recovery"] is not None and r["recovery"] >= M.MIN_RECOVERY,
        "f_area": (r["rep_area_m2"] or 0) >= M.MIN_REP_AREA,
        "f_urban": not (URBAN_RE.search(r["name"]) or "도시형" in (r["kapt_type"] or "")),
    }
    if have_households:
        hh = r["households"]
        f["f_households"] = hh is not None and hh >= M.MIN_HOUSEHOLDS
        f["f_turnover"] = hh is not None and hh > 0 and r["vol_12m"] >= hh * M.MIN_TURNOVER
    return f


def scores(r: dict, income_pct: dict[str, float]) -> dict:
    s = {}
    s["loc"] = bin_score(r["dong_pct"], M.LOC_PCT_BINS) if r["dong_pct"] is not None else 1
    s["defense"] = bin_score(r["from_peak"], M.DEFENSE_BINS) if r["from_peak"] is not None else 1
    ip = income_pct.get(r["sgg"])
    s["income"] = bin_score(ip, M.INCOME_PCT_BINS) if ip is not None else 3
    s["liquidity"] = bin_score(r["vol_bear"], M.LIQ_BINS)
    age = (M.ASOF_YEAR - r["build_year"]) if r["build_year"] else 99
    s["age"] = le_score(age, M.AGE_YEARS)
    s["budget"] = le_score(r["price84"], M.BUDGET_BINS) if r["price84"] else 1
    s["dist"] = le_score(r["dist_km"], M.DIST_KM) if r["dist_km"] is not None else 1
    total = sum(s[k] * w for k, w in M.WEIGHTS.items()) / sum(M.WEIGHTS.values())
    out = {f"s_{k}": v for k, v in s.items()}
    out["score"] = round(total, 3)
    out["score100"] = round(total / 5 * 100, 1)
    out["income_missing"] = ip is None
    return out


def run(dirs: list[Path], out_dir: Path, households_csv: Path, income_csv: Path) -> dict:
    complexes = [c for c in load_all(dirs) if in_universe(c.sgg)]
    hh_rows = read_csv(households_csv)
    households = {r["cid"]: r for r in hh_rows}
    have_hh = bool(hh_rows)

    inc_rows = read_csv(income_csv)
    income = {r["sgg"]: float(r["income_per_capita"]) for r in inc_rows
              if r.get("income_per_capita", "").strip()}
    income_pct = percentiles(income) if income else {}

    rows = compute(complexes, households, income)
    add_dong_percentile(rows)
    for r in rows:
        r.update(filters(r, have_hh))
        r.update(scores(r, income_pct))
        keys = [k for k in r if k.startswith("f_")]
        r["pass_all"] = all(r[k] for k in keys)

    fixed = ["cid", "gu", "dong", "name", "build_year", "rep_area_m2", "households",
             "score", "score100", "recovery", "from_peak", "price84", "dist_km",
             "vol_12m", "vol_bear", "dong_ppa", "dong_pct"]
    all_cols = fixed + [k for k in rows[0] if k not in fixed] if rows else fixed
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "scorecard_all.csv", rows, all_cols)
    passed = sorted((r for r in rows if r["pass_all"]), key=lambda r: -r["score"])
    for i, r in enumerate(passed, 1):
        r["rank"] = i
    name = "scorecard.csv" if have_hh else "scorecard_preliminary.csv"
    stale = out_dir / ("scorecard_preliminary.csv" if have_hh else "scorecard.csv")
    if stale.exists():
        stale.unlink()
    _write(out_dir / name, passed, ["rank"] + all_cols)
    return {"complexes": len(rows), "passed": len(passed), "output": name,
            "households_applied": have_hh, "income_applied": bool(income),
            "income_missing_rows": sum(1 for r in rows if r["income_missing"])}


def _write(path: Path, rows: list[dict], cols: list[str]) -> None:
    def fmt(v):
        if isinstance(v, float):
            return round(v, 4)
        return v
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in cols})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="house_pjt 의 web/public/data/gu")
    ap.add_argument("--extra-dir", type=Path, action="append", default=[],
                    help="추가 CSR 폴더 (기본: data/dongtan/gu 가 있으면 포함)")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--households", type=Path, default=DATA / "households.csv")
    ap.add_argument("--income", type=Path, default=DATA / "inputs" / "gu_income.csv")
    a = ap.parse_args(argv)

    dirs = [a.data_dir, *a.extra_dir]
    dt = DATA / "dongtan" / "gu"
    if dt.exists() and dt not in dirs:
        dirs.append(dt)
    res = run(dirs, a.out, a.households, a.income)
    print(f"단지 {res['complexes']:,}개 → 필터 통과 {res['passed']:,}개 → data/{res['output']}")
    if not res["households_applied"]:
        print("[주의] households.csv 없음: 세대수·거래회전 필터 미적용 (예비 결과). "
              "python -m apt_analysis.kapt 실행 후 재실행하세요.")
    if not res["income_applied"]:
        print("[주의] gu_income.csv 에 소득값이 없음: 생활수준 점수를 모두 중립(3)으로 처리했습니다.")
    elif res["income_missing_rows"]:
        print(f"[주의] 소득 데이터가 없는 단지 {res['income_missing_rows']:,}개는 생활수준 3점 처리.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
