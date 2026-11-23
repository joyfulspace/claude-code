"""폭락 시점 신호 평가(과적합 허용): 고점 전후 ±b% 구간 안에 신호가 있으면 적중.   python3 backtest/run_crash_timing.py [밴드(기본 0.08)]   (CT_START=2001-01-02 로 시작일 변경)

지표별로 (창, 상위 q%)를 격자 탐색해 F1(= 적중 구간 비율 R 과 구간 안 신호 비율 P 의 조화평균)을 최대화한다.
우연 기준선: (1) 같은 신호 수를 무작위로 시간 이동했을 때 적중 수, (2) 가짜 지표(백분위 무작위 이동)로 같은 탐색을 했을 때 최고 F1.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import crash_window as cw  # noqa: E402
from run_avoid_opt import roll_win  # noqa: E402
from run_param_opt import build_specs  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
QG = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40]
COOL = 20
START = pd.Timestamp(os.environ.get("CT_START", "2006-01-02"))
SUF = "" if START.year == 2006 else f"_{START.year}"


def metrics(cond, wins, start_i, first_valid):
    ev = cw.events_from_cond(cond, start_i, COOL)
    w_ok = [w for w in wins if w[2] >= first_valid]                       # 지표가 계산 가능했던 시기의 구간만 평가
    if len(ev) == 0 or not w_ok:
        return None
    hit, inwin, first = cw.score_events(ev, w_ok)
    R = hit / len(w_ok); Pn = inwin / len(ev)
    F = 2 * R * Pn / (R + Pn) if R + Pn > 0 else 0.0
    return {"hit": hit, "nwin": len(w_ok), "N": len(ev), "inwin": inwin, "R": R, "P": Pn, "F1": F, "first": first, "ev": ev, "w_ok": w_ok}


def main():
    BAND = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
    pd.options.display.float_format = "{:.3f}".format; pd.options.display.width = 250
    k, ser, funds = load_all(); c = k["close"]; idx = k.index; n = len(c)
    start_i = int(idx.searchsorted(START))
    tops = [t for t in cw.zigzag_tops(c, 0.15) if t >= start_i]
    wins = cw.make_windows(c, tops, BAND)
    specs = build_specs(k, ser, funds); pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    rng = np.random.default_rng(0)
    print(f"폭락 고점 {len(tops)}개 | 밴드 ±{BAND:.0%} | 구간 길이 중앙값 {np.median([e - s for s, t, e in wins]):.0f}거래일 | 신호 간격 {COOL}거래일\n")
    rows, best = [], {}
    for (grp, nm), (pn, wl, dflt, fn) in specs.items():
        wl = [w for w in wl if np.isfinite(pct[(nm, w)]).any()]
        if not wl: continue
        cand = []
        for w in wl:
            arr = np.nan_to_num(pct[(nm, w)], nan=-1)
            fv_w = int(np.flatnonzero(np.isfinite(pct[(nm, w)])).min())
            for q in QG:
                m = metrics(arr >= 1 - q / 100, wins, start_i, fv_w)
                if m: cand.append((m["F1"], m["hit"], -m["N"], w, q, m))
        if not cand: continue
        F, hit, negN, w, q, m = max(cand, key=lambda x: (x[0], x[1], x[2]))
        best[nm] = (grp, w, q, m)
        # (1) 같은 신호를 무작위로 시간 이동
        arr = np.nan_to_num(pct[(nm, w)], nan=-1); cond = arr >= 1 - q / 100
        fv_w = int(np.flatnonzero(np.isfinite(pct[(nm, w)])).min())
        rh, rf = [], []
        for _ in range(300):
            mm = metrics(roll_win(cond, start_i, rng.integers(252, n - start_i - 252)), wins, start_i, fv_w)
            rh.append(mm["hit"] if mm else 0); rf.append(mm["F1"] if mm else 0)
        # (2) 가짜 지표 탐색 (백분위를 창마다 독립적으로 무작위 이동, 같은 격자)
        nb = []
        for _ in range(100):
            bt = 0.0
            for w2 in wl:
                fa = np.nan_to_num(roll_win(pct[(nm, w2)], start_i, rng.integers(252, n - start_i - 252)), nan=-1)
                for q2 in QG:
                    mm = metrics(fa >= 1 - q2 / 100, wins, start_i, fv_w)
                    if mm and mm["F1"] > bt: bt = mm["F1"]
            nb.append(bt)
        rows.append({"그룹": grp, "지표": nm, "창": w, "상위%": q, "적중": f"{m['hit']}/{m['nwin']}", "hit": m["hit"], "신호": m["N"], "구간내신호": m["inwin"], "재현율R": m["R"], "정밀도P": m["P"], "F1": m["F1"],
                     "무작위 평균적중": np.mean(rh), "p(적중)": float(np.mean(np.array(rh) >= m["hit"])), "탐색null F1 평균": np.mean(nb), "탐색null F1 95%": np.percentile(nb, 95)})
    T = pd.DataFrame(rows).sort_values(["F1", "hit"], ascending=False).reset_index(drop=True); T.index += 1
    print("[지표별 최적(과적합 포함) — F1 순]")
    print(T[["그룹", "지표", "창", "상위%", "적중", "신호", "구간내신호", "재현율R", "정밀도P", "F1", "무작위 평균적중", "p(적중)", "탐색null F1 95%"]].to_string())
    pd.to_pickle({"T": T, "best": best, "tops": tops, "wins": wins, "band": BAND}, OUT / f"crash_timing_{int(BAND * 100)}{SUF}.pkl")


if __name__ == "__main__":
    main()
