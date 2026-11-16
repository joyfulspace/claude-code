"""지표별 신호 발생 시점 시각화(한글): python3 backtest/explain_signal_timeline.py  (먼저 run_avoid_opt.py)
파라미터 = λ=0.1 에서 최적화된 (창, 상위 q%)"""
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
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
GC = {"수급": "#2b8a3e", "시장민감도": "#1971c2", "심리": "#f08c00"}
LAM = 0.1


def owners(S, ev):
    own = {}
    for i in ev:
        c = [(abs(i - pk), j) for j, (pk, tr, _, _) in enumerate(S.ep) if max(pk - PRE, S.start_i) <= i <= tr]
        if c:
            own[i] = min(c)[1]
    first = {}
    for i in ev:
        j = own.get(i)
        if j is not None and j not in first:
            first[j] = i
    return set(own), set(first.values())


def main():
    k, ser, funds = load_all(); S = Scorer(k); c = k["close"]; idx = k.index
    R = pd.read_pickle(OUT / "avoid_opt_results.pkl"); pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    T = R[R["λ"] == LAM].copy(); T["검증"] = (T["기간교차"] + T["구간별교차"]) / 2
    T = T.sort_values("검증", ascending=False).reset_index(drop=True)
    sigs = {}
    for _, r in T.iterrows():
        ev = S.events(pct[(r["지표"], int(r["창"]))] >= 1 - r["상위q%"] / 100)
        inw, cred = owners(S, ev)
        sigs[r["지표"]] = (ev, inw, cred)

    def draw(t0, t1, fname, title, linear):
        fig, axes = plt.subplots(2, 1, figsize=(19, 12.5), sharex=True, gridspec_kw={"height_ratios": [2.3, 6]})
        a0, a1 = axes
        m = (idx >= t0) & (idx <= t1)
        (a0.plot if linear else a0.semilogy)(idx[m], c[m], color="#1f3b73", lw=1.4)
        for pk, tr, pkd, depth in S.ep:
            for a in axes:
                a.axvspan(idx[pk], idx[tr], color="#e03131", alpha=0.10)
            if t0 <= idx[pk] <= t1:
                a0.annotate(f"{depth:.0%}", (idx[pk], c.iloc[pk]), textcoords="offset points", xytext=(0, 8), color="#e03131", fontsize=11, fontweight="bold", ha="center")
        a0.set_title(title); a0.set_ylabel("KOSPI"); a0.grid(alpha=0.25)
        n = len(T)
        for row, (_, r) in enumerate(T.iterrows()):
            y = n - 1 - row
            ev, inw, cred = sigs[r["지표"]]
            col = GC[r["그룹"]]
            ev_m = [i for i in ev if t0 <= idx[i] <= t1]
            out = [i for i in ev_m if i not in inw]; ins = [i for i in ev_m if i in inw and i not in cred]; cr = [i for i in ev_m if i in cred]
            if out: a1.scatter(idx[out], [y] * len(out), marker="v", s=55, facecolors="white", edgecolors=col, linewidths=1.5, zorder=3)
            if ins: a1.scatter(idx[ins], [y] * len(ins), marker="v", s=55, color=col, alpha=0.65, zorder=3)
            if cr: a1.scatter(idx[cr], [y] * len(cr), marker="v", s=170, color=col, edgecolors="k", linewidths=1.2, zorder=4)
            a1.axhline(y, color="#e9ecef", lw=0.8, zorder=0)
        a1.set_yticks(range(n))
        a1.set_yticklabels([f"{r['지표']} (창 {int(r['창'])}, 상위 {int(r['상위q%'])}%, {len(sigs[r['지표']][0])}번)" for _, r in T.iloc[::-1].iterrows()], fontsize=11)
        for lab, (_, r) in zip(a1.get_yticklabels(), T.iloc[::-1].iterrows()):
            lab.set_color(GC[r["그룹"]])
        a1.set_xlim(t0, t1); a1.grid(alpha=0.2, axis="x")
        a1.set_title("지표별 매도 신호 발생 시점 (▼=신호, 빨간 음영=큰 하락 구간, 글자색=지표 그룹)")
        from matplotlib.lines import Line2D
        a1.legend(handles=[Line2D([], [], marker="v", ls="", color="k", markersize=13, label="큰 하락마다 처음 인정된 신호 (피한 손실 계산에 쓰임)"),
                           Line2D([], [], marker="v", ls="", color="#555", alpha=0.65, markersize=9, label="큰 하락 구간 안의 다른 신호"),
                           Line2D([], [], marker="v", ls="", markerfacecolor="white", markeredgecolor="#555", markersize=9, label="큰 하락과 무관하게 울린 신호 (기회비용만 발생)")],
                  loc="lower left", fontsize=10.5, framealpha=0.95)
        a1.xaxis.set_major_locator(mdates.YearLocator(1) if not linear else mdates.MonthLocator(interval=2))
        a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if not linear else "%y.%m"))
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=110); plt.close(fig)

    draw(idx[S.start_i], idx[-1], "sig_1_전체시계열.png", "KOSPI와 큰 하락 구간 (2006~, 로그 스케일)", False)
    draw(pd.Timestamp("2025-06-01"), idx[-1], "sig_2_최근시계열.png", "최근 상황 확대 (2025-06~): 2026-06-22 고점 이후 신호", True)

    print("최근 신호(마지막 3번):")
    for _, r in T.iterrows():
        ev = sigs[r["지표"]][0]
        print(f"  {r['지표']:<12} " + ", ".join(str(idx[i].date()) for i in ev[-3:]))
    cur_peak = c.loc["2026-01-01":].idxmax()
    print(f"\n2026-06-22 고점 이후 처음 울린 신호:")
    for _, r in T.iterrows():
        ev = sigs[r["지표"]][0]; after = [i for i in ev if idx[i] >= cur_peak - pd.Timedelta(days=90)]
        print(f"  {r['지표']:<12} " + (", ".join(f"{idx[i].date()}({c.iloc[i] / c.loc[:idx[i]].max() - 1:.0%})" for i in after[:3]) if after else "없음"))


if __name__ == "__main__":
    main()
