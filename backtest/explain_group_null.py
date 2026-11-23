"""'그룹 2개 이상'과 '우연 기준선' 원리 설명 그림(한글): python3 backtest/explain_group_null.py  (먼저 run_combo.py)"""
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
from run_avoid_opt import sc  # noqa: E402
from run_combo import event_arrays, roll_win  # noqa: E402
from run_param_opt import build_specs  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
GC = {"수급": "#2b8a3e", "시장민감도": "#1971c2", "심리": "#f08c00"}
RED = "#e03131"


def main():
    k, ser, funds = load_all(); S = Scorer(k); c = k["close"]; idx = k.index
    specs = build_specs(k, ser, funds); pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    names = [nm for (g, nm) in specs]; group = {nm: g for (g, nm) in specs}; dflt = {nm: d for (g, nm), (pn, w, d, fn) in specs.items()}
    EV = event_arrays(S, [pct[(nm, dflt[nm])] >= 0.8 for nm in names])
    D = pd.read_pickle(OUT / "combo_results.pkl"); ge = D["ge"]
    W = 20

    # ================= 그림 1: 그룹 2개 이상이란?
    t0, t1 = pd.Timestamp("2026-02-15"), pd.Timestamp("2026-07-31")
    m = (idx >= t0) & (idx <= t1); ii = np.flatnonzero(m)
    fig, axes = plt.subplots(5, 1, figsize=(17, 12.5), sharex=True, gridspec_kw={"height_ratios": [2.3, 2, 2, 2, 1.7]})
    axes[0].plot(idx[m], c[m], color="#1f3b73", lw=1.6); axes[0].set_ylabel("KOSPI"); axes[0].grid(alpha=0.25)
    for pk, tr, _, depth in S.ep:
        if idx[tr] >= t0 and idx[pk] <= t1: axes[0].axvspan(max(idx[pk], t0), min(idx[tr], t1), color=RED, alpha=0.10)
    axes[0].set_title("① 그룹 하나가 '울렸다' = 그 그룹 지표 중 하나라도 최근 20거래일 안에 신호(▼)가 있었다  →  ② 울린 그룹이 2개 이상이면 '진짜 신호'")
    active = {}
    for ax, g in zip(axes[1:4], ("수급", "시장민감도", "심리")):
        members = [nm for nm in names if group[nm] == g]
        act = np.zeros(S.n, bool)
        for r_, nm in enumerate(members):
            evs = np.flatnonzero(EV[names.index(nm)]); evs = evs[(idx[evs] >= t0) & (idx[evs] <= t1)]
            ax.scatter(idx[evs], [r_] * len(evs), marker="v", s=90, color=GC[g], edgecolors="k", linewidths=0.8, zorder=4)
            for i in evs:
                ax.hlines(r_, idx[i], idx[min(i + W, S.n - 1)], color=GC[g], lw=5, alpha=0.30)
                act[i:i + W] = True
            ax.text(t0, r_, nm + "  ", ha="right", va="center", fontsize=10.5, color=GC[g])
        active[g] = act
        ax.set_yticks([]); ax.set_ylim(-0.7, len(members) - 0.3)
        ax.axvspan(t0, t0, color="white")
        ax.set_ylabel(f"{g} 그룹", color=GC[g], fontweight="bold", rotation=0, ha="right", va="center", labelpad=95)
        ax.fill_between(idx[ii], -0.7, len(members) - 0.3, where=act[ii], color=GC[g], alpha=0.10, step="post")
        ax.grid(alpha=0.2, axis="x")
    cnt = sum(active[g].astype(int) for g in active)
    ax = axes[4]
    ax.step(idx[ii], cnt[ii], where="post", color="k", lw=2.5)
    ax.fill_between(idx[ii], 0, cnt[ii], where=(cnt[ii] >= 2), step="post", color=RED, alpha=0.30)
    ax.axhline(2, color=RED, ls="--", lw=1.2); ax.set_ylim(-0.1, 3.4); ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("울린 그룹 수", rotation=0, ha="right", va="center", labelpad=60)
    first = [i for i in S.events(cnt >= 2) if t0 <= idx[i] <= t1]
    for i in first:
        ax.scatter([idx[i]], [cnt[i]], marker="v", s=260, color=RED, edgecolors="k", zorder=6)
        ax.annotate(f"진짜 신호! ({idx[i].date()})\n울린 그룹 {int(cnt[i])}개", (idx[i], cnt[i]), textcoords="offset points", xytext=(25, 8), fontweight="bold", color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.grid(alpha=0.2, axis="x"); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
    fig.text(0.5, 0.005, "굵은 막대 = 신호 후 20거래일 동안 '울린 상태'. 같은 그룹 안에서 여러 지표가 울려도 1개 그룹으로만 셉니다.", ha="center", fontsize=12)
    fig.tight_layout(rect=(0.06, 0.02, 1, 1)); fig.savefig(OUT / "why_1_그룹2개이상.png", dpi=110); plt.close(fig)

    # ================= 그림 2: 우연 기준선 원리
    allE = np.ones(len(S.ep), bool); LAM = 0.1
    def score_of(cond):
        d = S.detail(cond); return sc(d, allE, np.ones(len(d["m"]), bool), LAM), d
    real_score, real_d = score_of(ge >= 2)
    rng = np.random.default_rng(3); n = S.n
    null = []
    for _ in range(400):
        null.append(score_of(roll_win(ge, S.start_i, rng.integers(252, n - S.start_i - 252)) >= 2)[0])
    null = np.array(null); p95 = np.percentile(null, 95)
    rng2 = np.random.default_rng(11); shifts = [int(rng2.integers(252, n - S.start_i - 252)) for _ in range(3)]
    fig = plt.figure(figsize=(19, 9.5)); gs = fig.add_gridspec(4, 2, width_ratios=[1.7, 1], hspace=0.55, wspace=0.25)
    rows = [("진짜 신호 (실제 날짜)", 0)] + [(f"가짜 {j + 1}: 신호를 통째로 {s_ / 252:.0f}년 옆으로 밀기(평가 기간 안에서 한 바퀴 돌림)", s_) for j, s_ in enumerate(shifts)]
    for r_, (lab, sh) in enumerate(rows):
        ax = fig.add_subplot(gs[r_, 0])
        ax.semilogy(idx, c, color="#1f3b73", lw=0.9)
        for pk, tr, _, _ in S.ep: ax.axvspan(idx[pk], idx[tr], color=RED, alpha=0.12)
        sc_, d = score_of(roll_win(ge, S.start_i, sh) >= 2) if sh else (real_score, real_d)
        ev = d["ev"]; ax.scatter(idx[ev], c.iloc[ev], marker="v", s=75, color="k" if sh == 0 else "#868e96", zorder=5)
        ax.set_xlim(idx[S.start_i - 60], idx[-1]); ax.set_title(f"{lab}  →  점수 {sc_:.3f}  (신호 {len(ev)}번, 큰 하락 {int((d['B'] > 0).sum())}/8 에서 인정)", fontsize=12.5, color="k" if sh == 0 else "#495057", loc="left")
        ax.set_yticks([]); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if r_ < 3: ax.set_xticklabels([])
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.hist(null, bins=24, color="#ced4da", edgecolor="white")
    ax2.axvline(p95, color="k", ls="--", lw=2.2); ax2.axvline(real_score, color=RED, lw=3.2)
    ymax = ax2.get_ylim()[1]
    ax2.text(p95, ymax * 0.97, " 우연 95% 선\n (400번 중 380번은\n 이보다 낮음)", fontsize=11.5, va="top", ha="right")
    ax2.text(real_score, ymax * 0.70, f" 진짜 점수 {real_score:.3f}", color=RED, fontsize=12.5, fontweight="bold", va="top")
    ax2.set_xlabel("점수 (λ=0.1)"); ax2.set_ylabel("가짜 신호 400번의 점수 분포(횟수)")
    ax2.set_title("신호를 400번 무작위로 옆으로 밀어서 얻은 점수 분포")
    fig.suptitle("우연 기준선의 원리: 신호의 '횟수와 간격'은 그대로 두고 '큰 하락과의 위치'만 무작위로 바꿔서 점수를 다시 잰다", fontsize=16, fontweight="bold", y=0.99)
    fig.savefig(OUT / "why_2_우연기준선원리.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"진짜 {real_score:.3f} | 우연 평균 {null.mean():.3f}, 95% {p95:.3f}, 진짜보다 높은 가짜 {(null >= real_score).sum()}개/400")


if __name__ == "__main__":
    main()
