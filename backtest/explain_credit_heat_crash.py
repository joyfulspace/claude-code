"""신용융자 과열 상세(폭락 시점 평가, -20% 기준): python3 backtest/explain_credit_heat_crash.py  (먼저 CT_START=2001-01-02 CT_DEV=0.20 run_crash_timing.py 0.08)"""
import sys, warnings
from pathlib import Path
import koreanize_matplotlib  # noqa: F401
import matplotlib; matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import crash_window as cw
from run_sell_lab import load_all
warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"; NM = "신용융자 과열(수준 또는 증가)"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREEN, ORANGE, GRAY = "#e03131", "#1971c2", "#2b8a3e", "#f08c00", "#868e96"

k, ser, funds = load_all(); c = k["close"]; idx = k.index
D = pd.read_pickle(OUT / "crash_timing_8_2001_d20.pkl"); grp, W, Q, m = D["best"][NM]; tops, tr, dr, wins = D["tops"], D["troughs"], D["drops"], D["wins"]
pct = pd.read_pickle(OUT / "param_opt_pct.pkl")[(NM, W)]; start_i = int(idx.searchsorted(pd.Timestamp("2001-01-02")))
thr = 1 - Q / 100; cond = np.nan_to_num(pct, nan=-1) >= thr; ev = cw.events_from_cond(cond, start_i, 20)
f = funds.reindex(idx).ffill().shift(1); ratio = f["margin"] / f["deposit"]; gr = ratio.pct_change(W)
fv = int(np.flatnonzero(np.isfinite(pct)).min())
print(f"설정: 증가창 {W}일, 상위 {Q}% (백분위 ≥ {thr:.0%}), 신호 {len(ev)}번, 계산 시작 {idx[fv].date()}")
# 폭락별 표
rows = []
for (s, t, e), b, d in zip(wins, tr, dr):
    inn = ev[(ev >= s) & (ev <= e)]; pre = ev[(ev < s) & (ev >= s - 60)]
    status = "n/a(데이터 이전)" if e < fv else ("적중" if len(inn) else "놓침")
    rows.append({"고점": idx[t].date(), "낙폭": f"{d:.0%}", "인정구간": f"{idx[s].date()}~{idx[e].date()}", "구간(거래일)": e - s, "결과": status,
                 "첫 신호 (고점대비 일)": (int(inn[0] - t) if len(inn) else None), "구간내 신호": len(inn), "저점까지 남은 일": (int(b - inn[0]) if len(inn) else None),
                 "고점 때 백분위": (round(float(pct[t]), 2) if np.isfinite(pct[t]) else None)})
T = pd.DataFrame(rows); pd.options.display.width = 250; print(T.to_string(index=False))
inw = np.zeros(len(ev), bool)
for s, t, e in wins: inw |= (ev >= s) & (ev <= e)
print(f"\n신호 {len(ev)}번 중 구간 안 {inw.sum()}번, 밖 {(~inw).sum()}번 (밖 신호 = 오경보)")
# 오경보 이후 60일 최대낙폭
fa = [(idx[i].date(), f"{c.iloc[i:i+61].min()/c.iloc[i]-1:.1%}") for i in ev[~inw]]
print("구간 밖 신호 → 이후 60일 최저 수익률:", fa)
T.to_csv(OUT / "credit_heat_crash_table.csv", index=False)
# 신호 날짜 CSV
pd.DataFrame({"신호일": idx[ev], "구간내": inw, "KOSPI": c.iloc[ev].values, "백분위": pct[ev]}).to_csv(OUT / "credit_heat_crash_signals.csv", index=False)

# 그림 1: 전체 시계열
fig, ax = plt.subplots(3, 1, figsize=(19, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.6]})
ax[0].semilogy(idx[start_i:], c.iloc[start_i:], color="#1f3b73", lw=1.3)
for (s, t, e), d in zip(wins, dr):
    for a in ax: a.axvspan(idx[s], idx[e], color=RED, alpha=0.18)
    ax[0].annotate(f"{d:.0%}", (idx[t], c.iloc[t]), textcoords="offset points", xytext=(0, 8), ha="center", color=RED, fontweight="bold", fontsize=10.5)
o = ev[~inw]; i_ = ev[inw]
ax[0].scatter(idx[o], c.iloc[o], marker="v", s=70, facecolors="white", edgecolors=BLUE, linewidths=1.8, zorder=4, label=f"신호 - 구간 밖 {len(o)}번(오경보)")
ax[0].scatter(idx[i_], c.iloc[i_], marker="v", s=90, color=BLUE, zorder=4, label=f"신호 - 폭락 구간 안 {len(i_)}번")
ax[0].legend(loc="upper left"); ax[0].set_title(f"KOSPI(로그)와 신용융자 과열 신호 (붉은 띠 = 고점 ±8% 폭락 인정구간, {len(wins)}개)"); ax[0].grid(alpha=0.2)
ax[1].plot(idx[start_i:], ratio.iloc[start_i:] * 100, color=GREEN, lw=1.3); ax[1].set_ylabel("신용융자/예탁금 (%)"); ax[1].set_title("원자료: 신용융자 / 투자자예탁금 (수준)"); ax[1].grid(alpha=0.2)
ax[2].plot(idx[start_i:], pct[start_i:], color=ORANGE, lw=1.1); ax[2].axhline(thr, color="k", ls="--", lw=1.3); ax[2].set_ylim(0, 1.02)
ax[2].fill_between(idx[start_i:], thr, 1.02, where=cond[start_i:], color=BLUE, alpha=0.25)
ax[2].set_title(f"과열 백분위 = max(수준 백분위, {W}일 증가율 백분위) · 점선 = 상위 {Q}% 문턱({thr:.0%})"); ax[2].grid(alpha=0.2)
ax[2].xaxis.set_major_locator(mdates.YearLocator(2)); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); fig.savefig(OUT / "credit_heat_1_전체.png", dpi=110); plt.close(fig)

# 그림 2: 폭락별 확대(최대 14개 격자)
n = len(wins); cols = 4; rws = int(np.ceil(n / cols))
fig, axs = plt.subplots(rws, cols, figsize=(20, 3.7 * rws)); axs = axs.ravel()
for j, ((s, t, e), d, b) in enumerate(zip(wins, dr, tr)):
    a = axs[j]; lo, hi = max(0, s - 90), min(len(c) - 1, b + 20); x = idx[lo:hi + 1]
    a.plot(x, c.iloc[lo:hi + 1], color="#1f3b73", lw=1.4); a.axvspan(idx[s], idx[e], color=RED, alpha=0.18); a.axvline(idx[t], color=RED, ls=":")
    sel = ev[(ev >= lo) & (ev <= hi)]
    a.scatter(idx[sel], c.iloc[sel], marker="v", s=80, color=BLUE, zorder=4)
    a2 = a.twinx(); a2.plot(x, pct[lo:hi + 1], color=ORANGE, lw=1, alpha=0.8); a2.axhline(thr, color="k", ls="--", lw=0.8); a2.set_ylim(0, 1.05); a2.set_yticks([])
    a.set_title(f"{idx[t].date()} 고점 ({d:.0%})" + (" · 적중" if ((ev >= s) & (ev <= e)).any() else (" · 놓침" if e >= fv else " · n/a")), fontsize=11.5)
    a.tick_params(labelsize=9); a.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
for a in axs[n:]: a.axis("off")
fig.suptitle("폭락별 확대: 빨간 띠 = 인정구간, 점선 = 고점, ▼ = 신호, 주황선 = 과열 백분위(검은 점선 = 문턱)", fontsize=15, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "credit_heat_2_폭락별.png", dpi=105); plt.close(fig)

# 그림 3: 선행일수 + 오경보 후 수익률
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
lab = [f"{r['고점']}\n{r['낙폭']}" for r in rows]; lead = [r["첫 신호 (고점대비 일)"] for r in rows]
cl = [(GRAY if r["결과"].startswith("n/a") else (GREEN if r["결과"] == "적중" else RED)) for r in rows]
ax[0].bar(range(n), [v if v is not None else 0 for v in lead], color=cl)
for j, (r, v) in enumerate(zip(rows, lead)): ax[0].text(j, (v or 0) + (2 if (v or 0) >= 0 else -2), r["결과"] if v is None else f"{v:+d}일", ha="center", va="bottom" if (v or 0) >= 0 else "top", fontsize=10)
ax[0].axhline(0, color="k", lw=1); ax[0].set_xticks(range(n)); ax[0].set_xticklabels(lab, fontsize=8.5)
ax[0].set_title("구간 안 첫 신호가 고점 대비 언제였나 (음수 = 고점 전, 양수 = 고점 후)"); ax[0].grid(alpha=0.25, axis="y")
fwd = [c.iloc[i:i + 61].min() / c.iloc[i] - 1 for i in ev]
ax[1].bar(range(len(ev)), np.array(fwd) * 100, color=[BLUE if x else "#adb5bd" for x in inw]); ax[1].set_xticks(range(len(ev))); ax[1].set_xticklabels([str(idx[i].date())[2:7] for i in ev], rotation=70, fontsize=8)
ax[1].set_title("각 신호 후 60거래일 안 최저점까지 수익률(%) (파랑 = 구간 안, 회색 = 구간 밖)"); ax[1].grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(OUT / "credit_heat_3_선행과_오경보.png", dpi=110); plt.close(fig)
print("완료")
