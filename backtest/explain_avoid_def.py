"""'하락을 잘 피했다' 점수 정의 그림(한글): python3 backtest/explain_avoid_def.py"""
import sys
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from avoid_score import Scorer, FWD, PRE  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, GREEN = "#e03131", "#2b8a3e"


def main():
    k, ser, funds = load_all(); S = Scorer(k); c = k["close"]; idx = k.index
    dd = (c / c.rolling(252, min_periods=60).max() - 1).values
    cond = dd <= -0.08; cond[:S.start_i] = False
    # 사례 1: 2007-10 하락 구간에서 처음 울린 신호
    pk, tr, pkd, depth = [e for e in S.ep if str(e[2].date()) == "2007-10-31"][0]
    lo = pk - PRE; i0 = lo + int(np.argmax(cond[lo:tr + 1])); P1 = S.o[i0 + 1]
    # 사례 2: 오경보 — 큰 하락 구간 밖에서 울렸고 이후 가장 많이 오른 신호
    inep = np.zeros(S.n, bool)
    for p_, t_, _, _ in S.ep: inep[max(p_ - PRE, 0):t_ + 1] = True
    ev = [i for i in S.events(cond) if not inep[i]]
    i1 = max(ev, key=lambda i: S.m[i]); P2 = S.o[i1 + 1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    # --- 성공 사례
    ax = axes[0]
    a, b = max(i0 - 100, 0), tr + 25
    ax.plot(idx[a:b], c.iloc[a:b], color="#1f3b73", lw=1.5)
    ax.axvspan(idx[i0 + 1], idx[min(i0 + FWD, S.n - 1)], color="#fff3bf", alpha=0.6, label="기회비용을 재는 구간(매도 후 120거래일)")
    ax.hlines(P1, idx[i0 + 1], idx[tr], color="k", ls="--", lw=1)
    ax.scatter([idx[i0 + 1]], [P1], color=RED, marker="v", s=220, zorder=6)
    ax.annotate(f"① 신호 발동 → 다음날 전량 매도 ({idx[i0 + 1].date()})", (idx[i0 + 1], P1), textcoords="offset points", xytext=(20, -75), fontweight="bold", arrowprops=dict(arrowstyle="->"), bbox=dict(boxstyle="round", fc="white", ec="#adb5bd"))
    B1, m1 = (P1 - S.c[tr]) / P1, S.m[i0]
    ax.annotate("", (idx[tr], S.c[tr]), (idx[tr], P1), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=3))
    ax.text(idx[tr], (P1 + S.c[tr]) / 2, f"② 피한 손실 {B1:.0%} ", color=GREEN, fontweight="bold", va="center", ha="right", fontsize=14, bbox=dict(boxstyle="round", fc="white", ec=GREEN))
    w = S.c[i0 + 1:i0 + 1 + FWD]; ip = i0 + 1 + int(np.argmax(w))
    ax.annotate("", (idx[ip], w.max()), (idx[ip], P1), arrowprops=dict(arrowstyle="<->", color=RED, lw=3))
    ax.text(idx[ip], w.max(), f"  ③ 놓친 상승 {m1:.0%}\n  (일찍 판 대가)", color=RED, fontweight="bold", va="top", ha="left", fontsize=14, bbox=dict(boxstyle="round", fc="white", ec=RED))
    ax.scatter([idx[pk]], [S.c[pk]], color="#f08c00", s=100, zorder=5); ax.annotate("고점", (idx[pk], S.c[pk]), textcoords="offset points", xytext=(-40, 6), color="#f08c00", fontweight="bold")
    ax.scatter([idx[tr]], [S.c[tr]], color="k", s=100, zorder=5); ax.annotate("저점", (idx[tr], S.c[tr]), textcoords="offset points", xytext=(10, -22), fontweight="bold")
    ax.set_title(f"사례 1) 큰 하락(고점 {idx[pk].date()}, {depth:.0%}) 구간에서 울린 신호")
    ax.legend(loc="lower left", fontsize=11); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m")); ax.grid(alpha=0.25)
    # --- 오경보 사례
    ax = axes[1]
    a, b = max(i1 - 100, 0), min(i1 + FWD + 40, S.n)
    ax.plot(idx[a:b], c.iloc[a:b], color="#1f3b73", lw=1.5)
    ax.axvspan(idx[i1 + 1], idx[min(i1 + FWD, S.n - 1)], color="#fff3bf", alpha=0.6, label="기회비용을 재는 구간(매도 후 120거래일)")
    ax.hlines(P2, idx[i1 + 1], idx[min(i1 + FWD, S.n - 1)], color="k", ls="--", lw=1)
    ax.scatter([idx[i1 + 1]], [P2], color=RED, marker="v", s=220, zorder=6)
    ax.annotate(f"① 신호 발동 → 다음날 전량 매도 ({idx[i1 + 1].date()})", (idx[i1 + 1], P2), textcoords="offset points", xytext=(15, -60), fontweight="bold", arrowprops=dict(arrowstyle="->"))
    w = S.c[i1 + 1:i1 + 1 + FWD]; ip = i1 + 1 + int(np.argmax(w))
    ax.annotate("", (idx[ip], w.max()), (idx[ip], P2), arrowprops=dict(arrowstyle="<->", color=RED, lw=3))
    ax.text(idx[ip], (w.max() + P2) / 2, f" ③ 놓친 상승 {S.m[i1]:.0%}\n (이 신호의 '기회비용')", color=RED, fontweight="bold", va="center", ha="right", fontsize=14, bbox=dict(boxstyle="round", fc="white", ec=RED))
    ax.set_title(f"사례 2) 큰 하락과 무관하게 울린 신호 (오경보) — 피한 손실은 없고 기회비용만 쌓임")
    ax.legend(loc="upper left", fontsize=11); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m")); ax.grid(alpha=0.25)
    fig.suptitle("점수 = (큰 하락 8번에서 평균 '피한 손실' ②) - λ × (모든 신호의 평균 '놓친 상승' ③)    ※ 규칙 예시: 고점 대비 -8%", fontsize=17, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "def_1_점수정의예시.png", dpi=110); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
