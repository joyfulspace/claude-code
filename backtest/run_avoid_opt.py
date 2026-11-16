"""지표별 파라미터 최적화(확정 정의) + 검증 + 비교.   python3 backtest/run_avoid_opt.py

목표: 점수(λ) = 큰 하락 8번의 평균 '피한 손실(금액형)' - λ × 모든 신호의 평균 '놓친 상승'   (λ = 0.25, 0.5, 1 각각 최적화)
탐색: 지표 창 길이 × 임계(상위 q% = 5,10,15,20,25,30), 방향은 이론대로 고정. 선택은 인접 임계 3개의 평균이 높은 곳(한 점 튐 방지).
검증: (1) 기간 교차(~2012 / 2013~)  (2) 구간별 교차(큰 하락 1개 빼고 찾아 그 하락에 적용)  (3) 우연 기준선(가짜 지표에 같은 절차)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from avoid_score import Scorer, PRE  # noqa: E402
from run_param_opt import build_specs  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
Q = [5, 10, 15, 20, 25, 30]
LAMS = [0.25, 0.5, 1.0]
SPLIT = pd.Timestamp("2013-01-01")
NFAKE = 150


class Ctx:
    def __init__(self, S):
        self.S = S
        self.nE = len(S.ep)
        self.peak_dates = np.array([e[2] for e in S.ep], dtype="datetime64[ns]")
        self.EA = self.peak_dates < np.datetime64(SPLIT)
        self.win = [(max(pk - PRE, S.start_i), tr) for pk, tr, _, _ in S.ep]


def sc(det, E, Vmask, lam):
    b = det["B"][E].mean()
    v = det["m"][Vmask] if len(det["m"]) else np.array([])
    return b - lam * (v.mean() if len(v) else 0.0)


def vmask(det, kind, ctx, j=None):
    ev, d = det["ev"], det["dates"]
    if len(ev) == 0:
        return np.zeros(0, bool)
    if kind == "all":
        return np.ones(len(ev), bool)
    if kind == "A":
        return np.asarray(d < SPLIT)
    if kind == "B":
        return np.asarray(d >= SPLIT)
    if kind == "loo":
        lo, hi = ctx.win[j]
        return ~((ev >= lo) & (ev <= hi))


def select(grid, E, kind, lam, ctx, j=None):
    """grid[w][q] = detail. 인접 q 3개 평균이 최대인 (w,q) 반환."""
    nw, nq = len(grid), len(grid[0])
    raw = np.array([[sc(grid[w][q], E, vmask(grid[w][q], kind, ctx, j), lam) for q in range(nq)] for w in range(nw)])
    sm = np.array([[np.mean(raw[w, max(q - 1, 0):q + 2]) for q in range(nq)] for w in range(nw)])
    return np.unravel_index(int(np.argmax(sm)), sm.shape), raw


def procedure(grid, ctx, lam):
    """전체 최적 + 기간교차 + 구간별교차 점수."""
    allE = np.ones(ctx.nE, bool)
    (w, q), raw = select(grid, allE, "all", lam, ctx)
    full = raw[w, q]
    # 기간 교차
    cv = []
    for tr_mask, tr_kind, te_mask, te_kind in ((ctx.EA, "A", ~ctx.EA, "B"), (~ctx.EA, "B", ctx.EA, "A")):
        (w2, q2), _ = select(grid, tr_mask, tr_kind, lam, ctx)
        d = grid[w2][q2]
        cv.append(sc(d, te_mask, vmask(d, te_kind, ctx), lam))
    # 구간별 교차
    loo = []
    for j in range(ctx.nE):
        E = np.ones(ctx.nE, bool); E[j] = False
        (w2, q2), _ = select(grid, E, "loo", lam, ctx, j)
        d = grid[w2][q2]
        mm = d["m"].mean() if len(d["m"]) else 0.0
        loo.append(d["B"][j] - lam * mm)
    return {"w": w, "q": q, "full": full, "cv": float(np.mean(cv)), "loo": float(np.mean(loo)), "raw": raw}


def main():
    pd.options.display.float_format = "{:.3f}".format; pd.options.display.width = 250
    k, ser, funds = load_all()
    S = Scorer(k); ctx = Ctx(S)
    specs = build_specs(k, ser, funds)
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    c = k["close"]
    print(f"큰 하락 {S.n and ctx.nE}개 | 기간교차: ~2012 {int(ctx.EA.sum())}개 / 2013~ {int((~ctx.EA).sum())}개 | 평가 시작 {S.idx[S.start_i].date()}\n")

    grids, info = {}, {}
    for (g, nm), (pn, wins, dflt, fn) in specs.items():
        grids[nm] = [[S.detail(pct[(nm, w)] >= 1 - q / 100) for q in Q] for w in wins]
        info[nm] = (g, pn, wins, dflt)

    # ---- 우연 기준선: 가짜 지표(백분위를 무작위로 시간 이동), 창 4개 탐색과 같은 자유도
    rng = np.random.default_rng(0); n = S.n
    keys = [(nm, info[nm][3]) for nm in grids]
    null = {lam: {"full": [], "cv": [], "loo": []} for lam in LAMS}
    for _ in range(NFAKE):
        arrs = [np.roll(pct[keys[rng.integers(len(keys))]], int(rng.integers(500, n - 500))) for _ in range(4)]
        g = [[S.detail(a >= 1 - q / 100) for q in Q] for a in arrs]
        for lam in LAMS:
            r = procedure(g, ctx, lam)
            for key in ("full", "cv", "loo"):
                null[lam][key].append(r[key])
    nl = {lam: {key: (np.mean(v), np.percentile(v, 95)) for key, v in d.items()} for lam, d in null.items()}

    rows = []
    for nm, grid in grids.items():
        g, pn, wins, dflt = info[nm]
        for lam in LAMS:
            r = procedure(grid, ctx, lam)
            d = grid[r["w"]][r["q"]]
            dd = grid[wins.index(dflt)][Q.index(20)]
            rows.append({"그룹": g, "지표": nm, "λ": lam, "파라미터명": pn, "창": wins[r["w"]], "상위q%": Q[r["q"]],
                         "기본값점수": sc(dd, np.ones(ctx.nE, bool), np.ones(len(dd["m"]), bool), lam),
                         "최적점수(전체)": r["full"], "기간교차": r["cv"], "구간별교차": r["loo"],
                         "신호": len(d["ev"]), "피한손실": d["B"].mean(), "놓친상승": d["m"].mean() if len(d["m"]) else 0.0,
                         "큰하락중신호": int((d["B"] > 0).sum()), "B_j": d["B"].tolist()})
    R = pd.DataFrame(rows)
    R.to_pickle(OUT / "avoid_opt_results.pkl")
    pd.to_pickle({"null": nl, "Q": Q, "grids_raw": {nm: {lam: procedure(grids[nm], ctx, lam)["raw"] for lam in LAMS} for nm in grids},
                  "info": info, "ep": [(str(e[2].date()), e[3]) for e in S.ep]}, OUT / "avoid_opt_aux.pkl")
    print("[우연 기준선] 가짜 지표 %d회 (평균 / 95%%)" % NFAKE)
    for lam in LAMS:
        print(f"  λ={lam}: 전체최적 {nl[lam]['full'][0]:.3f}/{nl[lam]['full'][1]:.3f} | 기간교차 {nl[lam]['cv'][0]:.3f}/{nl[lam]['cv'][1]:.3f} | 구간별교차 {nl[lam]['loo'][0]:.3f}/{nl[lam]['loo'][1]:.3f}")
    print()
    for lam in LAMS:
        T = R[R["λ"] == lam].copy()
        T["검증평균"] = (T["기간교차"] + T["구간별교차"]) / 2
        T = T.sort_values("검증평균", ascending=False)
        print(f"[λ={lam}] 지표 비교 (최적값 기준; 우연 95%: 전체 {nl[lam]['full'][1]:.3f} / 기간 {nl[lam]['cv'][1]:.3f} / 구간별 {nl[lam]['loo'][1]:.3f})")
        print(T[["그룹", "지표", "창", "상위q%", "기본값점수", "최적점수(전체)", "기간교차", "구간별교차", "신호", "피한손실", "놓친상승", "큰하락중신호"]].to_string(index=False), "\n")
    # 순위 안정성
    rk = pd.DataFrame({lam: R[R["λ"] == lam].set_index("지표").apply(lambda r: (r["기간교차"] + r["구간별교차"]) / 2, axis=1).rank(ascending=False) for lam in LAMS})
    print("[λ별 검증 순위] (1=가장 좋음)"); print(rk.sort_values(0.5).to_string())


if __name__ == "__main__":
    main()
