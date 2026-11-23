"""지표 조합 평가 그림(한글): python3 backtest/explain_combo.py  (먼저 run_combo.py)"""
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
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREEN, ORANGE, GRAY = "#e03131", "#1971c2", "#2b8a3e", "#f08c00", "#868e96"


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
    D = pd.read_pickle(OUT / "combo_results.pkl"); res = D["res"]; ns = D["ns"]; W0 = D["W0"]
    NS = 8

    # ---- 그림 1: n 에 따른 변화 (W=20)
    r = res[W0]; b = r["base"].iloc[:NS]; x = np.arange(1, NS + 1)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
    ax[0].bar(x - 0.2, b["신호"], 0.4, color="#adb5bd", label="신호 횟수(20년)"); ax[0].bar(x + 0.2, b["큰하락중신호"], 0.4, color=RED, label="큰 하락 8번 중 신호를 잡은 수")
    for xi, v, w in zip(x, b["신호"], b["큰하락중신호"]): ax[0].text(xi - 0.2, v, int(v), ha="center", va="bottom", fontsize=10.5); ax[0].text(xi + 0.2, w, int(w), ha="center", va="bottom", fontsize=10.5, color=RED)
    ax[0].set_title("n이 커질수록 신호가 급격히 줄어요"); ax[0].legend(fontsize=11)
    ax[1].plot(x, b["피한손실"] * 100, marker="o", lw=2.5, color=GREEN, label="평균 피한 손실 (큰 하락마다, %)")
    ax[1].plot(x, b["총기회비용"] * 100, marker="s", lw=2.5, color=ORANGE, label="총 기회비용 (큰 하락 1번당, %)")
    ax[1].set_title("피한 손실도, 기회비용도 함께 줄어요"); ax[1].legend(fontsize=11)
    sc_ = b["점수λ=0.1"].astype(float).values
    ax[2].fill_between(x, r["nullm"][0.1][:NS], r["null95"][0.1][:NS], color="#ced4da", alpha=0.7, label="우연 기준선 (평균~95%)")
    ax[2].plot(x, r["null95"][0.1][:NS], color=GRAY, ls="--", lw=1.5)
    ax[2].plot(x, sc_, marker="o", lw=3, color=BLUE, label="실제 점수 (λ=0.1)")
    for xi, v in zip(x, sc_): ax[2].text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=10.5, color=BLUE)
    ax[2].set_title("점수: 회색 띠 위로 나와야 '우연 이상'"); ax[2].legend(fontsize=11, loc="upper right")
    for a in ax: a.set_xlabel(f"n: 최근 {W0}거래일 안에 신호가 울린 지표 수가 n개 이상"); a.set_xticks(x); a.grid(alpha=0.25, axis="y")
    fig.suptitle(f"조합 평가 (기본값 지표 16개, 최근 {W0}거래일 기준)", fontsize=17, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "combo_1_n별변화.png", dpi=110); plt.close(fig)

    # ---- 그림 2: 창 W 별 점수 vs 우연 기준선
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8), sharey=True)
    for a, W in zip(ax, (5, 20, 60)):
        rr = res[W]; bb = rr["base"].iloc[:10]; xx = np.arange(1, 11)
        a.fill_between(xx, rr["nullm"][0.1][:10], rr["null95"][0.1][:10], color="#ced4da", alpha=0.7, label="우연 기준선(평균~95%)")
        a.plot(xx, bb["점수λ=0.1"], marker="o", lw=2.8, color=BLUE, label="실제 점수 (λ=0.1)")
        for xi, v, n_ in zip(xx, bb["점수λ=0.1"], bb["신호"]): a.text(xi, v - 0.025, f"{int(n_)}번", ha="center", fontsize=9.5, color=GRAY)
        a.set_title(f"최근 {W}거래일 안에 울린 지표 수 ≥ n"); a.set_xlabel("n"); a.set_xticks(xx); a.grid(alpha=0.25, axis="y")
    ax[0].set_ylabel("점수 (λ=0.1)"); ax[0].legend(fontsize=11, loc="lower left")
    fig.suptitle("'동시에 보는 기간'을 바꿔도: 파란 선이 회색 띠를 확실히 넘는 곳이 거의 없음 (점 아래 숫자 = 신호 횟수)", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "combo_2_창별.png", dpi=110); plt.close(fig)

    # ---- 그림 3: 파라미터 교차검증 (지표 n개 이상)
    fig, ax = plt.subplots(figsize=(11, 6.3))
    xs = np.arange(1, 9)
    ax.plot(xs, D["opt"]["점수λ=0.1"].iloc[:8].astype(float), marker="^", lw=2, color=GRAY, ls=":", label="전체구간 최적 파라미터 (낙관적)")
    ax.plot(xs, D["cv"][0.1][:8], marker="o", lw=2.8, color=GREEN, label="기간 교차검증 (훈련에서 고른 값 → 다른 기간)")
    ax.plot(xs, D["loo"][0.1][:8], marker="s", lw=2.8, color=ORANGE, label="구간별 교차검증 (하락 1개 빼고 고른 값)")
    ax.plot(xs, res[W0]["null95"][0.1][:8], "k--", lw=1.8, label="우연 95% 기준선")
    ax.set_xlabel(f"n: 최근 {W0}거래일 안에 신호가 울린 지표 수가 n개 이상"); ax.set_ylabel("점수 (λ=0.1)"); ax.set_xticks(xs); ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=10.5)
    ax.set_title("지표별 파라미터를 훈련 기간에서 고르면 점수가 낮아지고 기준선에 못 미침")
    fig.tight_layout(); fig.savefig(OUT / "combo_3_교차검증.png", dpi=110); plt.close(fig)

    # ---- 그림 4: 큰 하락별 피한 손실
    rows, labs = [], []
    for n in (1, 2, 3, 4, 5, 6):
        rows.append(res[W0]["base"].loc[n, "B_j"]); labs.append(f"지표 {n}개 이상 (신호 {int(res[W0]['base'].loc[n, '신호'])}번)")
    M = np.array([np.asarray(r_, float) for r_ in rows])
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.imshow(M, cmap="Greens", vmin=0, vmax=0.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "못 피함" if M[i, j] == 0 else f"{M[i, j]:.0%}", ha="center", va="center", fontsize=12.5, color="#868e96" if M[i, j] == 0 else ("white" if M[i, j] > 0.3 else "black"), fontweight="bold")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{d[:7]} 고점\n전체 {x_:.0%} 하락" for d, x_ in D["ep"]], fontsize=11)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=11.5)
    ax.set_title("조합 신호가 큰 하락 8번에서 피한 손실 (매도한 돈의 몇 %가 더 빠지는 걸 피했나)", fontsize=15, fontweight="bold", pad=12)
    for s_ in ax.spines.values(): s_.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "combo_4_구간별.png", dpi=110); plt.close(fig)

    # ---- 그림 5/6: 신호 시점 (조합 n=1,2,3,4,5)
    series = [(f"지표 {n}개 이상", res[W0]["cnt"] >= n) for n in (1, 2, 3, 4, 5)] 
    colors = [BLUE] * 5

    def draw(t0, t1, fname, title, linear):
        fig, axes = plt.subplots(2, 1, figsize=(19, 8.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 3.2]}); a0, a1 = axes
        m = (idx >= t0) & (idx <= t1)
        (a0.plot if linear else a0.semilogy)(idx[m], c[m], color="#1f3b73", lw=1.4)
        for pk, tr, pkd, depth in S.ep:
            for a in axes: a.axvspan(idx[pk], idx[tr], color=RED, alpha=0.10)
            if t0 <= idx[pk] <= t1: a0.annotate(f"{depth:.0%}", (idx[pk], c.iloc[pk]), textcoords="offset points", xytext=(0, 8), color=RED, fontsize=11, fontweight="bold", ha="center")
        a0.set_title(title); a0.set_ylabel("KOSPI"); a0.grid(alpha=0.25)
        nser = len(series)
        for row, ((lab, cond), col) in enumerate(zip(series, colors)):
            y = nser - 1 - row; ev = S.events(cond); inw, cred = owners(S, ev)
            evm = [i for i in ev if t0 <= idx[i] <= t1]
            out = [i for i in evm if i not in inw]; ins = [i for i in evm if i in inw and i not in cred]; cr = [i for i in evm if i in cred]
            if out: a1.scatter(idx[out], [y] * len(out), marker="v", s=60, facecolors="white", edgecolors=col, linewidths=1.6, zorder=3)
            if ins: a1.scatter(idx[ins], [y] * len(ins), marker="v", s=60, color=col, alpha=0.65, zorder=3)
            if cr: a1.scatter(idx[cr], [y] * len(cr), marker="v", s=190, color=col, edgecolors="k", linewidths=1.2, zorder=4)
            a1.axhline(y, color="#e9ecef", lw=0.8, zorder=0)
        a1.set_yticks(range(nser)); a1.set_yticklabels([f"{lab} ({len(S.events(cond))}번)" for lab, cond in series[::-1]], fontsize=11.5)
        a1.set_xlim(t0, t1); a1.grid(alpha=0.2, axis="x")
        a1.set_title(f"조합 신호 발생 시점 (최근 {W0}거래일 기준, ▼=신호, 큰 ▼=큰 하락마다 처음 인정, 빈 ▼=큰 하락과 무관)")
        a1.xaxis.set_major_locator(mdates.YearLocator(1) if not linear else mdates.MonthLocator(interval=2)); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if not linear else "%y.%m"))
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=110); plt.close(fig)
    draw(idx[S.start_i], idx[-1], "combo_5_전체시계열.png", "KOSPI와 조합 신호 (2006~)", False)
    draw(pd.Timestamp("2025-06-01"), idx[-1], "combo_6_최근시계열.png", "최근 상황 확대 (2025-06~)", True)
    print("완료")


if __name__ == "__main__":
    main()
