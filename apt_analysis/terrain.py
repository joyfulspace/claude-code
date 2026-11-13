"""단지 주변 지형(경사) 지표. SRTM 1초 타일(AWS elevation-tiles-prod, skadi)을 받아 로컬 계산한다.

  python -m apt_analysis.terrain --data-dir <house_pjt>/web/public/data/gu
산출: data/terrain.csv (cid, lat, lng, elev_m, slope500_pct, terrain_note)
※ SRTM 은 지표면+건물 높이(DSM)라 도심에서 다소 거칠게 나온다 → 백분위 상대평가에만 사용.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .common import load_all
from .scorecard import in_universe

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "dem"
URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{ns}{lat:02d}/{ns}{lat:02d}{ew}{lng:03d}.hgt.gz"
N = 3601                       # 1 arc-second
RADIUS_M = 500


def tile(lat: int, lng: int) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    name = f"N{lat:02d}E{lng:03d}.hgt"
    f = CACHE / name
    if not f.exists():
        url = URL.format(ns="N", lat=lat, ew="E", lng=lng)
        with urllib.request.urlopen(url, timeout=120) as r:
            f.write_bytes(gzip.decompress(r.read()))
    a = np.frombuffer(f.read_bytes(), dtype=">i2").reshape(N, N).astype(np.float32)
    a[a < -1000] = np.nan
    return a


def slope_pct(a: np.ndarray, lat0: int) -> np.ndarray:
    """픽셀별 경사(%). 1초 ≈ 위도 30.9m, 경도 30.9m*cos(lat)."""
    dy = 30.87
    dx = 30.87 * math.cos(math.radians(lat0 + 0.5))
    gy, gx = np.gradient(a, dy, dx)
    return np.hypot(gx, gy) * 100


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "terrain.csv")
    a = ap.parse_args(argv)
    cs = [c for c in load_all([a.data_dir]) if in_universe(c.sgg) and c.lat and c.lng]
    tiles, slopes = {}, {}
    rows = []
    r_px = int(RADIUS_M / 30.87)
    for c in cs:
        key = (int(math.floor(c.lat)), int(math.floor(c.lng)))
        if key not in tiles:
            try:
                tiles[key] = tile(*key)
            except Exception as e:                     # noqa: BLE001
                print(f"[경고] 타일 {key} 실패: {e}", file=sys.stderr)
                tiles[key] = None
            if tiles[key] is not None:
                slopes[key] = slope_pct(tiles[key], key[0])
        if tiles[key] is None:
            continue
        row = (1 - (c.lat - key[0])) * (N - 1)
        col = (c.lng - key[1]) * (N - 1)
        i, j = int(round(row)), int(round(col))
        i0, i1, j0, j1 = max(i - r_px, 0), min(i + r_px + 1, N), max(j - r_px, 0), min(j + r_px + 1, N)
        yy, xx = np.ogrid[i0:i1, j0:j1]
        mask = (yy - i) ** 2 + (xx - j) ** 2 <= r_px ** 2
        sl = slopes[key][i0:i1, j0:j1][mask]
        rows.append({"cid": c.cid, "lat": round(c.lat, 6), "lng": round(c.lng, 6),
                     "elev_m": round(float(tiles[key][i, j]), 1) if not np.isnan(tiles[key][i, j]) else "",
                     "slope500_pct": round(float(np.nanmean(sl)), 2) if sl.size else ""})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cid", "lat", "lng", "elev_m", "slope500_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"지형 지표 {len(rows):,}개 단지 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
