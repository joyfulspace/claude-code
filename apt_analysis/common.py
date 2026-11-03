"""공통: CSR(단지×월) 로더, 월 인덱스, 지표 계산.

CSR 형식(house_pjt export.py 참고):
  ids[i]            단지 ID ('11110-123')
  off[i]..off[i+1]  i번째 단지의 셀 구간
  dt[]              이전 셀로부터의 월 델타 (단지 첫 셀은 base 로부터)
  p[] n[] amt[]     평당가(만원/평, 전용) · 3개월 풀링 거래수 · 대표평형 금액(만원)
  meta[i]           [단지명, 동, 준공연도, 생애 거래수, lat, lng]
월 인덱스 = year*12 + (month-1).  base 는 2006-01.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

PYEONG_PER_M2 = 1 / 3.305785


def ym_to_idx(ym: str) -> int:
    s = ym.replace("-", "")
    return int(s[:4]) * 12 + int(s[4:6]) - 1


def idx_to_ym(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


@dataclass
class Complex:
    cid: str
    sgg: str
    gu_name: str
    name: str
    dong: str
    build_year: int
    life_trades: int
    lat: float
    lng: float
    t: list[int]
    p: list[float]
    n: list[int]
    amt: list[int]
    rep_area: float | None = None      # 대표 평형 전용면적(㎡)
    extra: dict = field(default_factory=dict)


def load_gu_file(path: Path) -> list[Complex]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    base, off = d["base"], d["off"]
    rep = _load_rep_areas(Path(path).with_name(f"{d['sgg']}.areas.json"))
    out = []
    for i, cid in enumerate(d["ids"]):
        a, b = off[i], off[i + 1]
        t, cur = [], base
        for k in range(a, b):
            cur += d["dt"][k]
            t.append(cur)
        m = d["meta"][i]
        out.append(Complex(
            cid=cid, sgg=d["sgg"], gu_name=d["name"], name=m[0], dong=m[1],
            build_year=m[2] or 0, life_trades=m[3],
            lat=m[4] or 0.0, lng=m[5] or 0.0,
            t=t, p=d["p"][a:b], n=d["n"][a:b], amt=d["amt"][a:b],
            rep_area=rep.get(cid),
        ))
    return out


def _load_rep_areas(path: Path) -> dict[str, float]:
    """areas.json 에서 단지별 대표 평형 = 거래건수가 가장 많은 면적 버킷."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    best: dict[str, tuple[int, float]] = {}
    for key, label in zip(d["ids"], d["labels"]):
        cid = key.split("/")[0]
        _lbl, ar, ntr = label
        if cid not in best or ntr > best[cid][0]:
            best[cid] = (ntr, ar)
    return {cid: ar for cid, (_n, ar) in best.items()}


def load_all(dirs: list[Path]) -> list[Complex]:
    seen, out = set(), []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name.endswith(".areas.json"):
                continue
            for c in load_gu_file(f):
                if c.cid not in seen:
                    seen.add(c.cid)
                    out.append(c)
    return out


# ── 지표 ───────────────────────────────────────────────
def anchor(c: Complex, T: int, tol: int = 2) -> float | None:
    """T 에서 ±tol 개월 내 가장 가까운 월의 p.
    같은 거리에 앞/뒤 월이 둘 다 있으면 거래수(n) 가중 평균."""
    best_d, cand = None, []
    for t, p, n in zip(c.t, c.p, c.n):
        d = abs(t - T)
        if d > tol:
            continue
        if best_d is None or d < best_d:
            best_d, cand = d, [(p, n)]
        elif d == best_d:
            cand.append((p, n))
    if not cand:
        return None
    w = sum(max(n, 1) for _p, n in cand)
    return sum(p * max(n, 1) for p, n in cand) / w


def peak(c: Complex, lo: int, hi: int) -> float | None:
    v = [p for t, p in zip(c.t, c.p) if lo <= t <= hi]
    return max(v) if v else None


def monthly_trades(c: Complex, lo: int, hi: int) -> float:
    """[lo,hi] 월평균 거래건수 추정.

    n 은 중심 3개월 풀링 건수라 한 거래가 최대 3번 세어진다 → Σn/3.
    (거래가 있는 달에만 셀이 있어 고립된 달은 약간 과소추정될 수 있다.)
    """
    months = hi - lo + 1
    return sum(n for t, n in zip(c.t, c.n) if lo <= t <= hi) / 3 / months


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
