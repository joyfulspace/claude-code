"""지표 조합 평가: 'n개 이상의 지표가 동시에 울리면 진짜 신호'.   python3 backtest/run_combo.py

점수 = 큰 하락 8번 평균 피한 손실 - λ × (모든 신호의 놓친 상승 합 / 큰 하락 수)   (avoid_score.py / run_avoid_opt.py 와 동일)
조합: count_t = (조건이 참인 지표 수), 신호 = count >= n 이 새로 참이 되는 날(120일 간격).  변형: 최근 W일 안의 발동 지표 수, 울린 '그룹' 수.
지표 파라미터: ① 기본값(창 기본, 상위20%)  ② 전체구간 최적(λ=0.1, 낙관적)  ③ 훈련구간에서 고른 값으로 검증구간 평가(교차) .
우연 기준선: 조합 count 시계열을 통째로 무작위 시간 이동(400회) -> 같은 n 에서 우연히 나오는 점수.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from avoid_score import Scorer  # noqa: E402
from run_avoid_opt import Q, LAMS, Ctx, sc, select, vmask  # noqa: E402
from run_param_opt import build_specs  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
NMAX = 16
NSH = 400


def event_arrays(S, conds):
    """지표마다 '신호 이벤트'(상승 에지, 120일 간격) 날짜를 1로 표시한 배열 (지표수, N)"""
    out = np.zeros((len(conds), S.n))
    for i, c in enumerate(conds):
        ev = S.events(np.asarray(c, bool))
        if len(ev): out[i, ev] = 1.0
    return out


def count_events(ev_arr, W):
    """최근 W거래일 안에 신호가 울린 지표 수"""
    a = pd.DataFrame(ev_arr.T).rolling(W, min_periods=1).max().values.T
    return a.sum(axis=0)


def eval_n(S, cnt, ns, ctx):
    rows = {}
    allE = np.ones(ctx.nE, bool)
    for n in ns:
        d = S.detail(cnt >= n)
        rows[n] = {"신호": len(d["ev"]), "큰하락중신호": int((d["B"] > 0).sum()), "피한손실": d["B"].mean(),
                   "총기회비용": d["m"].sum() / ctx.nE if len(d["m"]) else 0.0, "B_j": d["B"],
                   **{f"점수λ={l}": sc(d, allE, np.ones(len(d["m"]), bool), l) for l in LAMS}}
    return pd.DataFrame(rows).T


def main():
    pd.options.display.float_format = "{:.3f}".format; pd.options.display.width = 250
    k, ser, funds = load_all(); S = Scorer(k); ctx = Ctx(S); N = S.n
    specs = build_specs(k, ser, funds)
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    R = pd.read_pickle(OUT / "avoid_opt_results.pkl")
    names = [nm for (g, nm) in specs]; group = {nm: g for (g, nm) in specs}
    info = {nm: (wins, dflt) for (g, nm), (pn, wins, dflt, fn) in specs.items()}
    grids = {nm: [[S.detail(pct[(nm, w)] >= 1 - q / 100) for q in Q] for w in info[nm][0]] for nm in names}
    ns = list(range(1, NMAX + 1)); LAM0 = 0.1; allE = np.ones(ctx.nE, bool); rng = np.random.default_rng(0)
    conds_def = [pct[(nm, info[nm][1])] >= 0.8 for nm in names]
    EV = event_arrays(S, conds_def)
    print(f"큰 하락 {ctx.nE}개 | 지표 {len(names)}개 | 지표별 신호 = 기본값(창 기본, 상위 20%) 이벤트, 지표당 평균 {EV.sum(axis=1).mean():.1f}번\n")

    def null_for(cnt):
        nl = {l: np.zeros((NSH, len(ns))) for l in LAMS}; nsg = np.zeros((NSH, len(ns)))
        for t in range(NSH):
            cr = np.roll(cnt, int(rng.integers(500, N - 500)))
            for j, n in enumerate(ns):
                d = S.detail(cr >= n); nsg[t, j] = len(d["ev"])
                for l in LAMS:
                    nl[l][t, j] = sc(d, allE, np.ones(len(d["m"]), bool), l)
        return {l: np.percentile(nl[l], 95, axis=0) for l in LAMS}, {l: nl[l].mean(axis=0) for l in LAMS}, nsg

    res = {}
    for W in (5, 20, 60):
        cnt = count_events(EV, W); base = eval_n(S, cnt, ns, ctx); n95, nmn, nsg = null_for(cnt)
        res[W] = {"cnt": cnt, "base": base, "null95": n95, "nullm": nmn, "nsig": nsg}
    W0 = 20; base, n95, nsg = res[W0]["base"], res[W0]["null95"], res[W0]["nsig"]
    show = base[["신호", "큰하락중신호", "피한손실", "총기회비용"] + [f"점수λ={l}" for l in LAMS]].copy()
    show["우연95%(λ=0.1)"] = n95[0.1]; show["우연평균 신호"] = nsg.mean(axis=0)
    print(f"[① 기본값] 최근 {W0}거래일 안에 신호가 울린 지표가 n개 이상 → 진짜 신호  (우연 기준선: 조합 신호를 통째로 무작위 이동 {NSH}회)")
    print(show.head(12).to_string(), "\n")
    M = pd.DataFrame({l: base[f"점수λ={l}"].astype(float).values - n95[l] for l in LAMS}, index=ns)
    print("[① n × λ: 점수 - 우연 95%] (양수 = 우연보다 확실히 나음)"); print(M.head(12).round(3).to_string(), "\n")
    print("[창 W 민감도] 점수(λ=0.1) / 우연95% / 신호 수")
    tab = pd.DataFrame({f"W={W} 점수": res[W]["base"]["점수λ=0.1"] for W in (5, 20, 60)} | {f"W={W} 신호": res[W]["base"]["신호"] for W in (5, 20, 60)} | {f"W={W} 우연95%": res[W]["null95"][0.1] for W in (5, 20, 60)})
    print(tab.head(12).to_string(), "\n")
    # 그룹 변형: 그룹 안에서 W일 내 신호가 울린 지표가 하나라도 있으면 그룹 발동
    ge = np.stack([np.max(np.stack([np.asarray(pd.DataFrame(EV[[names.index(nm)]].T).rolling(W0, min_periods=1).max().values.T)[0] for nm in names if group[nm] == g]), axis=0) for g in ("수급", "시장민감도", "심리")]).sum(axis=0)
    gdf = eval_n(S, ge, [1, 2, 3], ctx)
    gn = {l: np.zeros((NSH, 3)) for l in LAMS}
    for t in range(NSH):
        cr = np.roll(ge, int(rng.integers(500, N - 500)))
        for j, g_ in enumerate((1, 2, 3)):
            d = S.detail(cr >= g_)
            for l in LAMS:
                gn[l][t, j] = sc(d, allE, np.ones(len(d["m"]), bool), l)
    g95 = {l: np.percentile(gn[l], 95, axis=0) for l in LAMS}
    gdf["우연95%(λ=0.1)"] = g95[0.1]
    print("[변형] 울린 '그룹' 수 ≥ g (수급/시장민감도/심리, 최근 20일)"); print(gdf[["신호", "큰하락중신호", "피한손실", "총기회비용"] + [f"점수λ={l}" for l in LAMS] + ["우연95%(λ=0.1)"]].to_string(), "\n")

    # ② 전체구간 최적(낙관적)
    T = R[R["λ"] == LAM0].set_index("지표")
    optc = [pct[(nm, int(T.loc[nm, "창"]))] >= 1 - T.loc[nm, "상위q%"] / 100 for nm in names]
    cnt_opt = count_events(event_arrays(S, optc), W0)
    opt = eval_n(S, cnt_opt, ns, ctx)

    # ③ 교차검증 (훈련 구간에서 지표별 파라미터 선택 → 검증 구간 평가)
    def pick_conds(E, kind, j=None):
        out = []
        for nm in names:
            (w, q), _ = select(grids[nm], E, kind, LAM0, ctx, j)
            out.append(pct[(nm, info[nm][0][w])] >= 1 - Q[q] / 100)
        return out
    cv = {l: np.zeros(len(ns)) for l in LAMS}; loo = {l: np.zeros(len(ns)) for l in LAMS}
    for tr_mask, tr_kind, te_mask, te_kind in ((ctx.EA, "A", ~ctx.EA, "B"), (~ctx.EA, "B", ctx.EA, "A")):
        cn = count_events(event_arrays(S, pick_conds(tr_mask, tr_kind)), W0)
        for j, n in enumerate(ns):
            d = S.detail(cn >= n)
            for l in LAMS:
                cv[l][j] += sc(d, te_mask, vmask(d, te_kind, ctx), l) / 2
    for jx in range(ctx.nE):
        E = np.ones(ctx.nE, bool); E[jx] = False
        cn = count_events(event_arrays(S, pick_conds(E, "loo", jx)), W0)
        for j, n in enumerate(ns):
            d = S.detail(cn >= n); mm = d["m"].sum() / ctx.nE if len(d["m"]) else 0.0
            for l in LAMS:
                loo[l][j] += (d["B"][jx] - l * mm) / ctx.nE
    cvt = pd.DataFrame({"②전체최적 신호": opt["신호"], "②전체최적 점수(낙관)": opt["점수λ=0.1"], "③기간교차 점수": cv[0.1], "③구간별교차 점수": loo[0.1],
                        "①기본값 점수": base["점수λ=0.1"], "①우연95%": n95[0.1]}, index=ns)
    print("[② 전체최적(낙관적) vs ③ 교차검증] 최근 20일 기준, 점수 λ=0.1"); print(cvt.head(12).to_string(), "\n")
    pd.to_pickle({"res": res, "ns": ns, "cv": cv, "loo": loo, "opt": opt, "gdf": gdf, "g95": g95, "ge": ge, "M": M, "names": names, "ep": [(str(e[2].date()), e[3]) for e in S.ep], "W0": W0}, OUT / "combo_results.pkl")


if __name__ == "__main__":
    main()
