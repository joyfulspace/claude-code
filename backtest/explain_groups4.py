"""4그룹(수급·시장민감도·심리·금리환율) '울린 그룹 수 ≥ g' 평가 그림(한글): python3 backtest/explain_groups4.py  (먼저 run_combo.py)"""
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
from avoid_score import Scorer, PRE  # noqa: E402
from run_combo import event_arrays  # noqa: E402
from run_param_opt import build_specs  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREEN, ORANGE, GRAY, PURPLE = "#e03131", "#1971c2", "#2b8a3e", "#f08c00", "#868e96", "#9c36b5"
GC = {"수급": GREEN, "시장민감도": BLUE, "심리": ORANGE, "금리·환율": PURPLE}
W = 20


def owners(S, ev):
    own = {}
    for i in ev:
        c = [(abs(i - pk), j) for j, (pk, tr, _, _) in enumerate(S.ep) if max(pk - PRE, S.start_i) <= i <= tr]
        if c: own[i] = min(c)[1]
    first = {}
    for i in ev:
        j = own.get(i)
        if j is not None and j not in first: first[j] = i
    return set(own), set(first.values())


def main():
    k, ser, funds = load_all(); S = Scorer(k); c = k["close"]; idx = k.index
    specs = build_specs(k, ser, funds); pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    D = pd.read_pickle(OUT / "combo_results.pkl"); ge = D["ge"]; groups = D["groups"]; gdf = D["gdf"]; g95 = D["g95"]; G = len(groups)
    names = [nm for (g, nm) in specs]; group = {nm: g for (g, nm) in specs}; dflt = {nm: d for (g, nm), (pn, w, d, fn) in specs.items()}
    EV = event_arrays(S, [pct[(nm, dflt[nm])] >= 0.8 for nm in names])
    gs = list(range(1, G + 1))
    dd_at = lambda i: c.iloc[i] / c.loc[:idx[i]].max() - 1

    # ---------- 그림 1: 4그룹 예시 (2026)
    t0, t1 = pd.Timestamp("2026-02-15"), pd.Timestamp("2026-07-31"); m = (idx >= t0) & (idx <= t1); ii = np.flatnonzero(m)
    fig, axes = plt.subplots(G + 2, 1, figsize=(17, 3.0 + 2.3 * G + 2), sharex=True, gridspec_kw={"height_ratios": [2.2] + [1.9] * G + [1.9]})
    axes[0].plot(idx[m], c[m], color="#1f3b73", lw=1.6); axes[0].set_ylabel("KOSPI"); axes[0].grid(alpha=0.25)
    for pk, tr, _, _ in S.ep:
        if idx[tr] >= t0 and idx[pk] <= t1:
            for a in axes: a.axvspan(max(idx[pk], t0), min(idx[tr], t1), color=RED, alpha=0.10)
    axes[0].set_title("그룹이 '울렸다' = 그 그룹 지표 중 하나라도 최근 20거래일 안에 신호(▼). 울린 그룹 수가 2, 3, 4개 이상이 되는 순간이 각각의 '진짜 신호'")
    act = {}
    for ax, g in zip(axes[1:G + 1], groups):
        members = [nm for nm in names if group[nm] == g]; a_ = np.zeros(S.n, bool)
        for r_, nm in enumerate(members):
            evs = np.flatnonzero(EV[names.index(nm)]); evs = evs[(idx[evs] >= t0) & (idx[evs] <= t1)]
            ax.scatter(idx[evs], [r_] * len(evs), marker="v", s=80, color=GC[g], edgecolors="k", linewidths=0.8, zorder=4)
            for i in evs:
                ax.hlines(r_, idx[i], idx[min(i + W, S.n - 1)], color=GC[g], lw=4.5, alpha=0.3); a_[i:i + W] = True
            ax.text(t0, r_, nm + "  ", ha="right", va="center", fontsize=9.5, color=GC[g])
        act[g] = a_; ax.set_yticks([]); ax.set_ylim(-0.7, len(members) - 0.3); ax.grid(alpha=0.2, axis="x")
        ax.fill_between(idx[ii], -0.7, len(members) - 0.3, where=a_[ii], color=GC[g], alpha=0.10, step="post")
        ax.set_ylabel(f"{g}", color=GC[g], fontweight="bold", rotation=0, ha="right", va="center", labelpad=105)
    cnt = sum(act[g].astype(int) for g in groups); ax = axes[-1]
    ax.step(idx[ii], cnt[ii], where="post", color="k", lw=2.4)
    for th, col in ((2, RED), (3, PURPLE), (4, "#111")):
        ax.axhline(th, color=col, ls="--", lw=1.1)
        for i in [i for i in S.events(cnt >= th) if t0 <= idx[i] <= t1]:
            ax.scatter([idx[i]], [cnt[i]], marker="v", s=230, color=col, edgecolors="k", zorder=6)
            ax.annotate(f"{th}그룹 이상 신호\n{idx[i].date()}", (idx[i], cnt[i]), textcoords="offset points", xytext=(18, 10), fontweight="bold", color=col, fontsize=11)
    ax.set_ylim(-0.1, G + 0.4); ax.set_yticks(range(G + 1)); ax.set_ylabel("울린 그룹 수", rotation=0, ha="right", va="center", labelpad=60)
    ax.text(0.99, 0.08, "이 기간에는 3그룹·4그룹 이상 신호가 없었음", transform=ax.transAxes, ha="right", fontsize=11, color=GRAY)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m")); ax.grid(alpha=0.2, axis="x")
    fig.tight_layout(rect=(0.05, 0, 1, 1)); fig.savefig(OUT / "g4_1_4그룹예시.png", dpi=105); plt.close(fig)

    # ---------- 그림 2: 그룹 수별 결과
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2)); x = np.arange(1, G + 1)
    ax[0].bar(x - 0.2, gdf["신호"], 0.4, color="#adb5bd", label="신호 횟수(20년)"); ax[0].bar(x + 0.2, gdf["큰하락중신호"], 0.4, color=RED, label="큰 하락 8번 중 잡은 수")
    for xi, v, w in zip(x, gdf["신호"], gdf["큰하락중신호"]): ax[0].text(xi - 0.2, v, int(v), ha="center", va="bottom"); ax[0].text(xi + 0.2, w, int(w), ha="center", va="bottom", color=RED)
    ax[0].set_title("울린 그룹이 많을수록 신호가 급감"); ax[0].legend(fontsize=11)
    lam_cols = [0.05, 0.1, 0.2, 0.5, 1.0]; cm = plt.cm.Blues(np.linspace(0.3, 0.95, len(lam_cols)))
    for i, l in enumerate(lam_cols):
        ax[1].plot(x, gdf[f"점수λ={l}"].astype(float), marker="o", lw=2.4, color=cm[i], label=f"λ={l}")
        ax[1].plot(x, g95[l], ls=":", marker="_", ms=16, color=cm[i], lw=1.2)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_title("점수(실선) vs 우연 95% 기준(점선) — 같은 색끼리 비교"); ax[1].legend(fontsize=10.5, ncol=2)
    M = np.array([[gdf[f"점수λ={l}"].astype(float).iloc[j] - g95[l][j] for l in lam_cols] for j in range(G)])
    im = ax[2].imshow(M, cmap="RdYlGn", vmin=-0.25, vmax=0.25, aspect="auto")
    for i in range(G):
        for j in range(len(lam_cols)): ax[2].text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax[2].set_xticks(range(len(lam_cols))); ax[2].set_xticklabels([f"λ={l}" for l in lam_cols]); ax[2].set_yticks(range(G)); ax[2].set_yticklabels([f"{g_}그룹 이상" for g_ in gs])
    ax[2].set_title("점수 - 우연 95% (초록=우연보다 나음)")
    for a in ax[:2]: a.set_xlabel("울린 그룹 수 ≥ g"); a.set_xticks(x); a.grid(alpha=0.25, axis="y")
    fig.suptitle(f"4그룹({' · '.join(groups)}) 기준, 최근 {W}거래일", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "g4_2_그룹수별결과.png", dpi=105); plt.close(fig)

    # ---------- 그림 3: 신호 시점의 고점 대비 낙폭
    fig, ax = plt.subplots(figsize=(15, 7.5)); rng = np.random.default_rng(0)
    for g_ in gs:
        ev = S.events(ge >= g_); inw, cred = owners(S, ev)
        d = np.array([dd_at(i) for i in ev]) * 100
        xs = g_ + rng.uniform(-0.18, 0.18, len(ev))
        colr = np.where([i in cred for i in ev], RED, np.where([i in inw for i in ev], ORANGE, GRAY))
        ax.scatter(xs, d, c=colr, s=np.where([i in cred for i in ev], 150, 70), edgecolors="k", linewidths=0.6, zorder=3)
        ax.hlines(np.median(d), g_ - 0.3, g_ + 0.3, color="k", lw=3); ax.text(g_ + 0.33, np.median(d), f"중앙값 {np.median(d):.0f}%", va="center", fontsize=11.5, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(gs); ax.set_xticklabels([f"{g_}그룹 이상\n({len(S.events(ge >= g_))}번)" for g_ in gs]); ax.grid(alpha=0.25, axis="y")
    ax.set_ylabel("신호가 난 날의 고점 대비 낙폭 (%)"); ax.set_title("요구하는 그룹 수를 높일수록 신호일 낙폭 중앙값이 깊어짐: -12% → -13% → -15% → -20%  (4그룹은 3번뿐)\n빨강=큰 하락마다 처음 인정, 주황=큰 하락 구간 안, 회색=무관", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "g4_3_신호시점낙폭.png", dpi=105); plt.close(fig)

    # ---------- 그림 4: 타임라인
    for fname, (a_, b_), linear, ttl in (("g4_4_전체시계열.png", (idx[S.start_i], idx[-1]), False, "KOSPI와 그룹 수별 신호 시점 (2006~)"), ("g4_5_최근시계열.png", (pd.Timestamp("2025-06-01"), idx[-1]), True, "최근 확대 (2025-06~)")):
        fig, axes = plt.subplots(2, 1, figsize=(19, 8.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 3.0]}); a0, a1 = axes
        mm = (idx >= a_) & (idx <= b_); (a0.plot if linear else a0.semilogy)(idx[mm], c[mm], color="#1f3b73", lw=1.4)
        for pk, tr, _, depth in S.ep:
            for a in axes: a.axvspan(idx[pk], idx[tr], color=RED, alpha=0.10)
            if a_ <= idx[pk] <= b_: a0.annotate(f"{depth:.0%}", (idx[pk], c.iloc[pk]), textcoords="offset points", xytext=(0, 8), color=RED, fontsize=11, fontweight="bold", ha="center")
        a0.set_title(ttl); a0.set_ylabel("KOSPI"); a0.grid(alpha=0.25)
        for row, g_ in enumerate(gs):
            y = G - 1 - row; ev = S.events(ge >= g_); inw, cred = owners(S, ev); evm = [i for i in ev if a_ <= idx[i] <= b_]
            out = [i for i in evm if i not in inw]; ins = [i for i in evm if i in inw and i not in cred]; cr = [i for i in evm if i in cred]
            if out: a1.scatter(idx[out], [y] * len(out), marker="v", s=60, facecolors="white", edgecolors=PURPLE, linewidths=1.6, zorder=3)
            if ins: a1.scatter(idx[ins], [y] * len(ins), marker="v", s=60, color=PURPLE, alpha=0.65, zorder=3)
            if cr: a1.scatter(idx[cr], [y] * len(cr), marker="v", s=190, color=PURPLE, edgecolors="k", linewidths=1.2, zorder=4)
            a1.axhline(y, color="#e9ecef", lw=0.8, zorder=0)
        a1.set_yticks(range(G)); a1.set_yticklabels([f"{g_}그룹 이상 ({len(S.events(ge >= g_))}번)" for g_ in gs[::-1]], fontsize=12); a1.set_xlim(a_, b_); a1.grid(alpha=0.2, axis="x")
        a1.set_title("▼=신호, 큰 ▼=큰 하락마다 처음 인정, 빈 ▼=큰 하락과 무관")
        a1.xaxis.set_major_locator(mdates.YearLocator(1) if not linear else mdates.MonthLocator(interval=2)); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if not linear else "%y.%m"))
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=105); plt.close(fig)

    # ---------- 그림 5: 큰 하락별 피한 손실
    Mx = np.array([np.asarray(gdf.loc[g_, "B_j"], float) for g_ in gs])
    fig, ax = plt.subplots(figsize=(16, 4.8)); ax.imshow(Mx, cmap="Greens", vmin=0, vmax=0.5, aspect="auto")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]): ax.text(j, i, "못 피함" if Mx[i, j] == 0 else f"{Mx[i, j]:.0%}", ha="center", va="center", fontsize=12.5, color="#868e96" if Mx[i, j] == 0 else ("white" if Mx[i, j] > 0.3 else "black"), fontweight="bold")
    ax.set_xticks(range(Mx.shape[1])); ax.set_xticklabels([f"{d[:7]} 고점\n전체 {x_:.0%} 하락" for d, x_ in D["ep"]], fontsize=11)
    ax.set_yticks(range(G)); ax.set_yticklabels([f"{g_}그룹 이상 (신호 {int(gdf.loc[g_, '신호'])}번)" for g_ in gs], fontsize=11.5)
    ax.set_title("그룹 수별 신호가 큰 하락 8번에서 피한 손실", fontsize=15, fontweight="bold", pad=12)
    for s_ in ax.spines.values(): s_.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "g4_6_구간별.png", dpi=105); plt.close(fig)
    print("완료")
    for g_ in gs:
        ev = S.events(ge >= g_); d = [dd_at(i) for i in ev]
        print(f"g>={g_}: 신호 {len(ev)}번, 신호일 낙폭 중앙값 {np.median(d):.0%}, 25~75% {np.percentile(d, 25):.0%}~{np.percentile(d, 75):.0%}")


if __name__ == "__main__":
    main()
