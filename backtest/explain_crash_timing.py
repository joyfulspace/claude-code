"""폭락 시점 신호 평가 그림(한글): python3 backtest/explain_crash_timing.py  (먼저 run_crash_timing.py 0.05/0.08/0.10)"""
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
import crash_window as cw  # noqa: E402
from run_crash_timing import COOL, START  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, GREEN, BLUE, ORANGE, GRAY = "#e03131", "#2b8a3e", "#1971c2", "#f08c00", "#868e96"
GC = {"수급": GREEN, "시장민감도": BLUE, "심리": ORANGE, "금리·환율": "#9c36b5"}


def main():
    k, ser, funds = load_all(); c = k["close"]; idx = k.index; n = len(c); start_i = int(idx.searchsorted(START))
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    D = {b: pd.read_pickle(OUT / f"crash_timing_{b}.pkl") for b in (5, 8, 10)}
    tops = D[8]["tops"]; W = {b: D[b]["wins"] for b in D}; T8 = D[8]["T"]; best8 = D[8]["best"]
    drop = []
    for t in tops:
        seg = c.values[t:t + 260]; drop.append(seg.min() / c.iloc[t] - 1)
    lab = [f"{idx[t].strftime('%y.%m')}\n({d:.0%})" for t, d in zip(tops, drop)]

    # ---- 그림 1: 폭락 고점과 구간
    fig = plt.figure(figsize=(19, 10)); gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0, :]); ax.semilogy(c.iloc[start_i:].index, c.iloc[start_i:], color="#1f3b73", lw=1.2)
    for (s, t, e), (s5, _, e5), d in zip(W[8], W[5], drop):
        ax.axvspan(idx[s], idx[e], color=RED, alpha=0.18); ax.axvspan(idx[s5], idx[e5], color=RED, alpha=0.35)
        ax.scatter([idx[t]], [c.iloc[t]], marker="v", s=70, color=RED, zorder=5); ax.annotate(f"{idx[t].strftime('%y.%m')}\n{d:.0%}", (idx[t], c.iloc[t]), textcoords="offset points", xytext=(0, 9), fontsize=9.5, ha="center", color=RED, fontweight="bold")
    ax.set_title(f"폭락 고점 {len(tops)}개 (종가 지그재그 15% 반전, 2006~). 진한 빨강=±5% 구간, 연한 빨강=±8% 구간(이번 평가 기본)"); ax.grid(alpha=0.25)
    for j, ti in enumerate((tops[2], tops[9], tops[13])):
        a = fig.add_subplot(gs[1, j]); t = ti; i0, i1 = max(t - 60, 0), min(t + 60, n - 1)
        a.plot(idx[i0:i1], c.iloc[i0:i1], color="#1f3b73", lw=1.5)
        for b, colr, al in ((8, RED, 0.18), (5, RED, 0.35)):
            s, _, e = [w for w in W[b] if w[1] == t][0]; a.axvspan(idx[s], idx[e], color=colr, alpha=al)
        s5, _, e5 = [w for w in W[5] if w[1] == t][0]
        a.scatter([idx[t]], [c.iloc[t]], marker="v", s=120, color=RED, zorder=5)
        a.hlines([c.iloc[t] / 1.05, c.iloc[t] * 0.95], idx[i0], idx[i1], color="k", ls=":", lw=1)
        a.text(0.02, 0.06, "구간 시작 = 고점으로 오르기 전\n마지막 5% 상승을 시작한 날\n구간 끝 = 고점에서 처음 5% 빠진 날" if j == 0 else "", transform=a.transAxes, fontsize=9.5, bbox=dict(boxstyle="round", fc="white", ec=GRAY))
        a.set_title(f"예시: {idx[t].strftime('%Y-%m-%d')} 고점 (±5%: {t - s5}일 전 ~ {e5 - t}일 후)"); a.grid(alpha=0.25); a.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
    fig.suptitle("'폭락 시점' 정의: 고점 전후 구간 안에서 신호가 나면 인정", fontsize=16, fontweight="bold"); fig.savefig(OUT / "crash_1_폭락구간정의.png", dpi=105, bbox_inches="tight"); plt.close(fig)

    # ---- 그림 2: 지표 순위 (±8%)
    T = T8.sort_values("F1"); y = np.arange(len(T))
    fig, ax = plt.subplots(1, 2, figsize=(19, 9.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].barh(y, T["F1"], color=[GC[g] for g in T["그룹"]]); ax[0].scatter(T["탐색null F1 95%"], y, marker="|", s=420, color="k", lw=3, zorder=5, label="우연 95% (가짜 지표로 같은 탐색)")
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{r['지표']} (창 {int(r['창'])}, 상위 {int(r['상위%'])}%)" for _, r in T.iterrows()], fontsize=11)
    for yi, (_, r) in zip(y, T.iterrows()): ax[0].text(0.395, yi, f"{r['적중']} 적중, 신호 {int(r['신호'])}번 중 {int(r['구간내신호'])}번 구간 안", va="center", fontsize=9.8)
    ax[0].set_xlim(0, 0.68); ax[0].set_xlabel("F1 (적중한 폭락 비율과 구간 안 신호 비율의 조화평균)"); ax[0].legend(loc="lower right", fontsize=10.5); ax[0].grid(alpha=0.25, axis="x")
    ax[0].set_title("지표별 최적 파라미터(과적합 포함)의 폭락 시점 신호 성능 — ±8% 구간")
    ax[1].barh(y, T["hit"], color="#74c0fc", label="적중한 폭락 수"); ax[1].scatter(T["무작위 평균적중"], y, marker="D", s=60, color=RED, zorder=5, label="같은 신호 수를 무작위로 찍었을 때 평균 적중")
    for yi, (_, r) in zip(y, T.iterrows()): ax[1].text(r["hit"] + 0.2, yi, f"p={r['p(적중)']:.3f}", va="center", fontsize=9.8)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlabel("적중한 폭락 수 (최대 15개)"); ax[1].legend(loc="lower right", fontsize=10.5); ax[1].grid(alpha=0.25, axis="x"); ax[1].set_title("무작위보다 많이 맞혔나? (p = 무작위가 같거나 더 많이 맞힐 확률)")
    fig.tight_layout(); fig.savefig(OUT / "crash_2_지표순위.png", dpi=105); plt.close(fig)

    # ---- 그림 3: 지표 x 폭락 히트맵 (첫 신호의 고점 대비 일수)
    order = list(T8.sort_values(["F1", "hit"], ascending=False)["지표"])
    M = np.full((len(order), len(tops)), np.nan); NA = np.zeros_like(M, bool)
    for r_, nm in enumerate(order):
        grp, w, q, m = best8[nm]; arr = np.nan_to_num(pct[(nm, w)], nan=-1); fv = int(np.flatnonzero(np.isfinite(pct[(nm, w)])).min())
        ev = cw.events_from_cond(arr >= 1 - q / 100, start_i, COOL)
        for j, (s, t, e) in enumerate(W[8]):
            if e < fv: NA[r_, j] = True; continue
            inn = ev[(ev >= s) & (ev <= e)]
            if len(inn): M[r_, j] = inn[0] - t
    fig, ax = plt.subplots(figsize=(19, 10.5)); cmap = plt.cm.Greens_r.copy(); cmap.set_bad("#f1f3f5")
    Z = np.ma.masked_invalid(np.abs(M)); ax.imshow(Z, cmap=cmap, vmin=0, vmax=90, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if NA[i, j]: ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#adb5bd")
            elif np.isnan(M[i, j]): ax.text(j, i, "·", ha="center", va="center", fontsize=14, color="#adb5bd")
            else: ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(len(tops))); ax.set_xticklabels(lab, fontsize=9.5); ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{nm} (창 {best8[nm][1]}, 상위 {best8[nm][2]}%)  {int(T8.set_index('지표').loc[nm, 'hit'])}/{int(T8.set_index('지표').loc[nm, '적중'].split('/')[1]) if False else T8.set_index('지표').loc[nm, '적중'].split('/')[1]}" for nm in order], fontsize=10.5)
    for lbl, nm in zip(ax.get_yticklabels(), order): lbl.set_color(GC[best8[nm][0]])
    ax.set_title("지표별 최적값이 각 폭락 구간에서 처음 신호를 낸 시점 (숫자 = 고점 대비 거래일: 음수=고점 전, 양수=고점 후 / 진할수록 고점에 가까움 / ·=못 맞힘 / n/a=계산 불가 시기)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "crash_3_지표x폭락.png", dpi=105); plt.close(fig)

    # ---- 그림 4: 상위 6개 지표 신호 시점
    top6 = list(T8.sort_values(["F1", "hit"], ascending=False)["지표"])[:6]
    fig, axes = plt.subplots(2, 1, figsize=(19, 8.6), sharex=True, gridspec_kw={"height_ratios": [2, 3]}); a0, a1 = axes
    cc = c.iloc[start_i:]; a0.semilogy(cc.index, cc, color="#1f3b73", lw=1.2)
    for s, t, e in W[8]:
        for a in axes: a.axvspan(idx[s], idx[e], color=RED, alpha=0.18)
        a0.scatter([idx[t]], [c.iloc[t]], marker="v", s=55, color=RED, zorder=5)
    a0.set_title("KOSPI와 폭락 구간(±8%)"); a0.grid(alpha=0.25)
    for r_, nm in enumerate(top6):
        yy = len(top6) - 1 - r_; grp, w, q, m = best8[nm]; ev = m["ev"]
        inn = np.zeros(len(ev), bool)
        for s, t, e in W[8]: inn |= (ev >= s) & (ev <= e)
        a1.scatter(idx[ev[~inn]], [yy] * int((~inn).sum()), marker="v", s=60, facecolors="white", edgecolors=GC[grp], linewidths=1.6, zorder=3)
        a1.scatter(idx[ev[inn]], [yy] * int(inn.sum()), marker="v", s=150, color=GC[grp], edgecolors="k", linewidths=1.0, zorder=4); a1.axhline(yy, color="#e9ecef", lw=0.8, zorder=0)
    a1.set_yticks(range(len(top6))); a1.set_yticklabels([f"{nm} (창 {best8[nm][1]}, 상위 {best8[nm][2]}%)\n{T8.set_index('지표').loc[nm, '적중']} 적중, 신호 {int(T8.set_index('지표').loc[nm, '신호'])}번" for nm in top6[::-1]], fontsize=11); a1.grid(alpha=0.2, axis="x")
    a1.set_title("F1 상위 6개 지표의 신호 (큰 채움 ▼ = 폭락 구간 안, 빈 ▼ = 구간 밖)"); a1.xaxis.set_major_locator(mdates.YearLocator(2)); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(OUT / "crash_4_상위지표신호.png", dpi=105); plt.close(fig)

    # ---- 그림 5: 구간 폭별 순위
    R = pd.DataFrame({f"±{b}%": D[b]["T"].set_index("지표")["F1"].rank(ascending=False) for b in D}); R["평균"] = R.mean(axis=1); R = R.sort_values("평균")
    H = pd.DataFrame({f"±{b}%": D[b]["T"].set_index("지표")["적중"] for b in D})
    fig, ax = plt.subplots(figsize=(11, 9.5)); Rv = R[["±5%", "±8%", "±10%"]].values
    ax.imshow(Rv, cmap="RdYlGn_r", vmin=1, vmax=20, aspect="auto")
    for i in range(Rv.shape[0]):
        for j in range(3): ax.text(j, i, f"{int(Rv[i, j])}위\n({H.loc[R.index[i]].iloc[j]})", ha="center", va="center", fontsize=9.8, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["±5% (중앙값 14일)", "±8% (35일, 기본)", "±10% (42일)"]); ax.set_yticks(range(len(R))); ax.set_yticklabels(R.index, fontsize=11)
    ax.set_title("구간 폭을 바꿔도 순위가 유지되나? (F1 순위, 괄호=적중 수)", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "crash_5_구간폭별순위.png", dpi=105); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
