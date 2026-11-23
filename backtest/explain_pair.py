"""예탁금 감소 + 한미 상관 조합 그림(한글): python3 backtest/explain_pair.py  (먼저 run_pair.py)"""
import sys
import warnings
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from avoid_score import Scorer  # noqa: E402
from explain_indicator import owners  # noqa: E402
from run_avoid_opt import LAMS, Q, Ctx, sc  # noqa: E402
from run_pair import CFGS, WB, make_cond, name_cfg  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
GREEN, RED, BLUE, ORANGE, GRAY, PURPLE = "#2b8a3e", "#e03131", "#1971c2", "#f08c00", "#868e96", "#9c36b5"


def main():
    k, ser, funds = load_all(); S = Scorer(k); ctx = Ctx(S); c = k["close"]; idx = k.index
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl"); F = pd.read_pickle(OUT / "pair_fixed.pkl"); P = pd.read_pickle(OUT / "pair_search.pkl")
    T1, dets = F["T1"], F["dets"]; names = list(T1.index)
    A = np.nan_to_num(pct[("예탁금 감소", 20)], nan=-1) >= 0.95
    allE = np.ones(ctx.nE, bool)
    colr = {n: (GREEN if "예탁금" in n and "AND" not in n and "OR" not in n else BLUE if "한미" in n else PURPLE) for n in names}

    # ---- 1: 점수 비교
    fig, ax = plt.subplots(1, 2, figsize=(19, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(len(names)); lam_list = [0.05, 0.1, 0.2]; wd = 0.26
    for i, l in enumerate(lam_list):
        ax[0].bar(x + (i - 1) * wd, T1[f"점수λ={l}"].astype(float), wd, color=plt.cm.Blues(0.35 + 0.25 * i), label=f"점수 λ={l}")
        ax[0].hlines(T1[f"우연95%λ={l}"].astype(float), x + (i - 1) * wd - wd / 2, x + (i - 1) * wd + wd / 2, color="k", ls="--", lw=1.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace("(", "\n(").replace(" 단독", "\n단독") for n in names], fontsize=9.8); ax[0].legend(fontsize=10.5, loc="lower left"); ax[0].grid(alpha=0.25, axis="y")
    ax[0].set_title("점수 비교 (같은 색 막대 위의 검은 점선 = 우연 95% 기준선)"); ax[0].set_ylim(0.1, 0.25)
    ax[1].bar(x - 0.2, T1["신호"].astype(float), 0.4, color="#adb5bd", label="신호 횟수"); ax[1].bar(x + 0.2, T1["큰하락중신호"].astype(float), 0.4, color=RED, label="큰 하락 8번 중 잡은 수")
    for xi, a_, b_ in zip(x, T1["신호"], T1["큰하락중신호"]): ax[1].text(xi - 0.2, a_, int(a_), ha="center", va="bottom", fontsize=9.5); ax[1].text(xi + 0.2, b_, int(b_), ha="center", va="bottom", color=RED, fontsize=9.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["예탁금\n단독", "한미\n단독", "OR", "AND\n같은날", "AND\n5일", "AND\n10일", "AND\n20일", "AND\n40일", "AND\n60일"], fontsize=9.5); ax[1].legend(fontsize=10.5); ax[1].set_title("신호 횟수와 큰 하락 포착"); ax[1].grid(alpha=0.25, axis="y")
    fig.suptitle("예탁금 감소(20일, 상위 5%) + 한미 상관(40일, 상위 30%) 을 같이 쓸 때", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "pair_1_점수비교.png", dpi=105); plt.close(fig)

    # ---- 2: 타임라인
    rows = ["예탁금 감소 단독(20일, 5%)", "한미 상관 단독(40일, 30%)", "OR(둘 중 하나)", "AND(최근 10일 안에)", "AND(같은 날)"]
    fig, axes = plt.subplots(2, 1, figsize=(19, 8.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 3.0]}); a0, a1 = axes
    cc = c.iloc[S.start_i:]; a0.semilogy(cc.index, cc, color="#1f3b73", lw=1.3)
    for pk, tr, _, depth in S.ep:
        for a in axes: a.axvspan(idx[pk], idx[tr], color=RED, alpha=0.10)
        a0.annotate(f"{depth:.0%}", (idx[pk], c.iloc[pk]), textcoords="offset points", xytext=(0, 8), color=RED, fontsize=10.5, fontweight="bold", ha="center")
    a0.set_title("KOSPI와 큰 하락 구간"); a0.grid(alpha=0.25)
    for r_, nm in enumerate(rows):
        y = len(rows) - 1 - r_; ev = dets[nm]["ev"]; own, first = owners(S, ev); fi = set(first.values())
        o = [i for i in ev if i not in own]; ins = [i for i in ev if i in own and i not in fi]; cr = list(fi); cc_ = colr[nm]
        if o: a1.scatter(idx[o], [y] * len(o), marker="v", s=60, facecolors="white", edgecolors=cc_, linewidths=1.6, zorder=3)
        if ins: a1.scatter(idx[ins], [y] * len(ins), marker="v", s=60, color=cc_, alpha=0.65, zorder=3)
        if cr: a1.scatter(idx[cr], [y] * len(cr), marker="v", s=190, color=cc_, edgecolors="k", linewidths=1.2, zorder=4)
        a1.axhline(y, color="#e9ecef", lw=0.8, zorder=0)
    a1.set_yticks(range(len(rows))); a1.set_yticklabels([f"{n} ({len(dets[n]['ev'])}번)" for n in rows[::-1]], fontsize=11.5); a1.grid(alpha=0.2, axis="x")
    a1.set_title("신호 시점 (큰 ▼=큰 하락마다 처음 인정, 작은 채움=큰 하락 구간 안, 빈 ▼=큰 하락과 무관)"); a1.xaxis.set_major_locator(mdates.YearLocator(2)); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(OUT / "pair_2_신호시점.png", dpi=105); plt.close(fig)

    # ---- 3: 구간별 피한 손실
    sel = rows; Mx = np.array([dets[n]["B"] for n in sel])
    fig, ax = plt.subplots(figsize=(16, 4.6)); ax.imshow(Mx, cmap="Greens", vmin=0, vmax=0.5, aspect="auto")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]): ax.text(j, i, "못 피함" if Mx[i, j] == 0 else f"{Mx[i, j]:.0%}", ha="center", va="center", fontsize=12.5, color="#868e96" if Mx[i, j] == 0 else ("white" if Mx[i, j] > 0.3 else "black"), fontweight="bold")
    ax.set_xticks(range(Mx.shape[1])); ax.set_xticklabels([f"{str(e[2].date())[:7]} 고점\n전체 {e[3]:.0%} 하락" for e in S.ep], fontsize=11); ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=11)
    ax.set_title("큰 하락 8번에서 피한 손실", fontsize=15, fontweight="bold", pad=12)
    for s_ in ax.spines.values(): s_.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "pair_3_구간별.png", dpi=105); plt.close(fig)

    # ---- 4: 한미 상관 파라미터 탐색 지형 (AND 최근 10일, 20일/40일 창 등)
    pB = {w: pct[("한미 상관", w)] for w in WB}
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    for a, cfg in zip(axes, (("AND", 10), ("AND", 1), ("OR", 0))):
        M = np.array([[sc(S.detail(make_cond(A, np.nan_to_num(pB[w], nan=-1) >= 1 - q / 100, cfg)), allE, np.ones(len(S.detail(make_cond(A, np.nan_to_num(pB[w], nan=-1) >= 1 - q / 100, cfg))["m"]), bool), 0.1) for q in Q] for w in WB])
        a.imshow(M, cmap="YlGn", vmin=0.15, vmax=0.23, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=11, fontweight="bold")
        a.set_xticks(range(len(Q))); a.set_xticklabels([f"{q}%" for q in Q]); a.set_yticks(range(len(WB))); a.set_yticklabels([f"{w}일" for w in WB])
        a.set_xlabel("한미 상관: 상위 몇 %일 때"); a.set_ylabel("한미 상관 계산 기간"); a.set_title(name_cfg(cfg))
    fig.suptitle("한미 상관 파라미터 탐색 지형 (점수 λ=0.1; 예탁금 감소 단독 = 0.216)", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "pair_4_탐색지형.png", dpi=105); plt.close(fig)

    # ---- 5: 최적 조합 vs 우연 기준선
    res, nl, one = P["res"], P["nl"], P["one"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8), sharey=False)
    for a, l in zip(ax, (0.05, 0.1, 0.2)):
        r = res[l]; vals = [r["full"], r["cv"], r["loo"]]; labs = ["전체 기간\n(과적합 포함)", "기간 교차검증", "구간별 교차검증"]
        a.bar(np.arange(3) - 0.2, vals, 0.4, color=[ "#74c0fc", GREEN, ORANGE], label="탐색한 최적 조합")
        a.bar(np.arange(3) + 0.2, [one[l]["full"], one[l]["cv"], one[l]["loo"]], 0.4, color="#ced4da", label="예탁금 감소 단독")
        for i, key in enumerate(("full", "cv", "loo")): a.hlines(nl[l][key][1], i - 0.45, i + 0.45, color="k", ls="--", lw=2, label="우연 95% (가짜 한미 상관)" if i == 0 else None)
        a.set_xticks(range(3)); a.set_xticklabels(labs, fontsize=10.5); a.set_title(f"λ={l}"); a.grid(alpha=0.25, axis="y"); a.set_ylim(0.1 if l <= 0.1 else 0.05, None)
    ax[0].legend(fontsize=10, loc="lower left"); fig.suptitle("한미 상관을 최적화해 붙인 조합 vs 예탁금 감소 단독 vs 우연 기준선", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "pair_5_검증.png", dpi=105); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
