"""지표 백분위 캐시 갱신(새 지표 추가·반대매매 구간 보정 반영): python3 backtest/build_pct_cache.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_param_opt import build_specs, ensure_cache  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

OUT = Path(__file__).parent / "results"
k, ser, funds = load_all()
specs = build_specs(k, ser, funds)
pct = ensure_cache(specs, OUT / "param_opt_pct.pkl")
print(f"지표 {len(specs)}개:")
for (g, nm), (pn, wins, d, fn) in specs.items():
    a = pct[(nm, d)]
    import numpy as np
    v = a[~np.isnan(a)]
    print(f"  [{g}] {nm:<26} 창 {wins} 기본 {d} | 유효 {len(v)}일 ({k.index[np.flatnonzero(~np.isnan(a))[0]].date() if len(v) else '-'}~)")
