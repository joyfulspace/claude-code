"""예탁금 감소 + 한미 상관 조합 평가와 한미 상관 파라미터 탐색(예탁금 감소 고정).   python3 backtest/run_pair.py

A = 예탁금 감소 (창 20, 상위 5%) 고정.  B = 한미 상관 (창 w, 상위 q%).
결합: OR (A 또는 B), AND(W) (A 이면서 최근 W거래일 안에 B 상태가 있었음; W=1 은 같은 날).
검증: 기간교차 / 구간별교차 / 우연 기준선(B 의 백분위를 무작위로 시간 이동해 같은 탐색을 반복).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from avoid_score import Scorer, PRE  # noqa: E402
from run_avoid_opt import LAMS, Q, Ctx, procedure, roll_win, sc  # noqa: E402
from run_param_opt import build_specs  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
B_NAME = sys.argv[1] if len(sys.argv) > 1 else "한미 상관"
SLUG = B_NAME.replace(" ", ""); SUF = "" if SLUG == "한미상관" else f"_{SLUG}"
WB = None  # main()에서 지표 정의의 창 목록으로 설정
CFGS = [("OR", 0)] + [("AND", w) for w in (1, 5, 10, 20, 40, 60)]
NFAKE = 150


def recent(b, W):
    if W <= 1:
        return b
    return pd.Series(b.astype(float)).rolling(W, min_periods=1).max().values > 0


def make_cond(A, B, cfg):
    kind, W = cfg
    return (A | B) if kind == "OR" else (A & recent(B, W))


def name_cfg(cfg):
    return "OR(둘 중 하나)" if cfg[0] == "OR" else (f"AND(같은 날)" if cfg[1] == 1 else f"AND(최근 {cfg[1]}일 안에)")


def main():
    pd.options.display.float_format = "{:.3f}".format; pd.options.display.width = 250
    global WB
    k, ser, funds = load_all(); S = Scorer(k); ctx = Ctx(S); N = S.n; idx = k.index; c = k["close"]
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl"); R = pd.read_pickle(OUT / "avoid_opt_results.pkl")
    specs = build_specs(k, ser, funds); WB = [v for (g_, n_), v in specs.items() if n_ == B_NAME][0][1]
    ro = R[(R["지표"] == B_NAME) & (R["λ"] == 0.1)].iloc[0]; WO, QO = int(ro["창"]), int(ro["상위q%"])
    A = np.nan_to_num(pct[("예탁금 감소", 20)], nan=-1) >= 0.95
    pB = {w: pct[(B_NAME, w)] for w in WB}
    Bopt = np.nan_to_num(pB[WO], nan=-1) >= 1 - QO / 100
    B_LAB = f"{B_NAME} 단독({WO}일, {QO}%)"
    allE = np.ones(ctx.nE, bool); rng = np.random.default_rng(0)

    def evalc(cond):
        d = S.detail(cond); return d, {l: sc(d, allE, np.ones(len(d["m"]), bool), l) for l in LAMS}
    # ====== 1) 최적 개별 파라미터로 같이 쓸 때
    rows, dets = {}, {}
    for name, cond in [("예탁금 감소 단독(20일, 5%)", A), (B_LAB, Bopt)] + [(name_cfg(cfg), make_cond(A, Bopt, cfg)) for cfg in CFGS]:
        d, s_ = evalc(cond); dets[name] = d
        nul = {l: np.percentile([sc(S.detail(roll_win(cond, S.start_i, rng.integers(252, N - S.start_i - 252))), allE, np.ones(0, bool), l) if False else 0 for _ in range(0)] or [0], 95) for l in LAMS}
        vals = {l: [] for l in LAMS}
        for _ in range(300):
            dd = S.detail(roll_win(cond, S.start_i, rng.integers(252, N - S.start_i - 252)))
            for l in LAMS: vals[l].append(sc(dd, allE, np.ones(len(dd["m"]), bool), l))
        rows[name] = {"신호": len(d["ev"]), "큰하락중신호": int((d["B"] > 0).sum()), "피한손실": d["B"].mean(), "총기회비용": d["m"].sum() / ctx.nE if len(d["m"]) else 0.0,
                      **{f"점수λ={l}": s_[l] for l in LAMS}, **{f"우연95%λ={l}": np.percentile(vals[l], 95) for l in LAMS}}
    T1 = pd.DataFrame(rows).T
    print(f"[1] 각자의 최적 파라미터로 같이 쓸 때 (예탁금 감소 20일/5% + {B_NAME} {WO}일/{QO}%)")
    print(T1[["신호", "큰하락중신호", "피한손실", "총기회비용", "점수λ=0.05", "점수λ=0.1", "점수λ=0.2", "우연95%λ=0.05", "우연95%λ=0.1", "우연95%λ=0.2"]].to_string(), "\n")
    pd.to_pickle({"T1": T1, "dets": dets, "B": B_NAME, "WO": WO, "QO": QO, "WB": WB}, OUT / f"pair_fixed{SUF}.pkl")

    # ====== 2) 한미 상관 파라미터 탐색 (A 고정)
    def build_grid(pBmap):
        grid, keys = [], []
        for cfg in CFGS:
            for w in WB:
                Bs = [np.nan_to_num(pBmap[w], nan=-1) >= 1 - q / 100 for q in Q]
                grid.append([S.detail(make_cond(A, b, cfg)) for b in Bs]); keys.append((cfg, w))
        return grid, keys
    grid, keys = build_grid(pB)
    res = {l: procedure(grid, ctx, l) for l in LAMS}
    print(f"[2] {B_NAME} 파라미터 탐색 결과 (예탁금 감소 고정, 결합방식·창·상위% 탐색)")
    for l in LAMS:
        r = res[l]; cfg, w = keys[r["w"]]; d = grid[r["w"]][r["q"]]
        print(f"  λ={l}: 최적 = {name_cfg(cfg)}, {B_NAME} 창 {w}일, 상위 {Q[r['q']]}% | 신호 {len(d['ev'])}번, 큰 하락 {int((d['B'] > 0).sum())}/8, 피한 {d['B'].mean():.3f}, 총비용 {d['m'].sum() / ctx.nE:.3f} | 점수(전체) {r['full']:.3f}, 기간교차 {r['cv']:.3f}, 구간별교차 {r['loo']:.3f}")
    # 우연 기준선: 한미 상관 백분위를 무작위 이동 (4개 창 각각 독립), 같은 탐색 반복
    null = {l: {"full": [], "cv": [], "loo": []} for l in LAMS}
    for t in range(NFAKE):
        pf = {w: roll_win(pB[w], S.start_i, rng.integers(252, N - S.start_i - 252)) for w in WB}
        g, _ = build_grid(pf)
        for l in LAMS:
            r = procedure(g, ctx, l)
            for key in ("full", "cv", "loo"): null[l][key].append(r[key])
    nl = {l: {key: (np.mean(v), np.percentile(v, 95)) for key, v in d.items()} for l, d in null.items()}
    print(f"\n  [우연 기준선: 가짜 {B_NAME}로 같은 탐색 {NFAKE}회] (평균 / 95%)")
    for l in LAMS: print(f"  λ={l}: 전체 {nl[l]['full'][0]:.3f}/{nl[l]['full'][1]:.3f} | 기간교차 {nl[l]['cv'][0]:.3f}/{nl[l]['cv'][1]:.3f} | 구간별교차 {nl[l]['loo'][0]:.3f}/{nl[l]['loo'][1]:.3f}")
    # 단독 비교: 예탁금 감소 단독 (같은 절차 CV)
    gA = [[S.detail(A)]]; one = {l: procedure([[S.detail(A)] * len(Q)], ctx, l) for l in LAMS}
    print("\n  [참고] 예탁금 감소 단독(파라미터 고정) 전체/기간교차/구간별교차:", {l: (round(one[l]['full'], 3), round(one[l]['cv'], 3), round(one[l]['loo'], 3)) for l in LAMS})
    # 상위 조합 목록 (λ=0.1, 전체 기간)
    lst = []
    for fi, (cfg, w) in enumerate(keys):
        for qi, q in enumerate(Q):
            d = grid[fi][qi]; lst.append({"결합": name_cfg(cfg), "B창": w, "상위%": q, "신호": len(d["ev"]), "큰하락": int((d["B"] > 0).sum()), "피한손실": d["B"].mean(), "점수λ=0.1": sc(d, allE, np.ones(len(d["m"]), bool), 0.1)})
    L = pd.DataFrame(lst).sort_values("점수λ=0.1", ascending=False)
    print("\n  [λ=0.1 상위 12개 조합 (전체 기간 점수, 과적합 포함)]"); print(L.head(12).to_string(index=False))
    # 필터 능력: 예탁금 감소 신호 중 이후 -10% 더 하락한 것 vs 아닌 것을 B 백분위로 구분하나?
    evA = [i for i in S.events(A) if S.wlen[i] >= 120]; valid = np.array([S.a[i] >= 0.10 for i in evA])
    def auc(x, y):
        r = pd.Series(x).rank(); n1 = y.sum(); n0 = len(y) - n1
        return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"\n  [필터 능력] 예탁금 감소 신호(완료 {len(evA)}번) 중 이후 -10% 더 하락 {int(valid.sum())}번")
    filt = {}
    for w in WB:
        x = np.array([pB[w][i] for i in evA]); ok = ~np.isnan(x)
        filt[w] = (float(np.nanmean(x[valid])), float(np.nanmean(x[~valid])), float(auc(x[ok], valid[ok])))
        print(f"    {B_NAME}({w}일) 백분위: 하락한 신호 평균 {filt[w][0]:.2f} vs 나머지 {filt[w][1]:.2f} | AUC {filt[w][2]:.2f}")
    pd.to_pickle({"grid_keys": keys, "res": res, "nl": nl, "one": one, "L": L, "filt": filt}, OUT / f"pair_search{SUF}.pkl")


if __name__ == "__main__":
    main()
