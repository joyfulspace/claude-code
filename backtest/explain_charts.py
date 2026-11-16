"""매도 신호 실험 결과를 한글 그림 5장으로 설명: python3 backtest/explain_charts.py
(먼저 run_sell_lab.py 를 실행해 results/sell_lab_*.csv 를 만들어 둘 것)"""
import sys
from pathlib import Path

import koreanize_matplotlib  # noqa: F401  (한글 폰트)
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
from run_decline_response import build_rules, events  # noqa: E402

OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, GREEN, BLUE, GRAY, ORANGE = "#e03131", "#2b8a3e", "#1971c2", "#adb5bd", "#f08c00"


def fig1(k, rules, dd):
    c, o = k["close"], k["open"]
    start = k.index[300]
    W = 250  # 매도 후 약 1년(거래일)을 보여준다
    def stats(d):
        i = c.index.get_loc(d); P = o.iloc[i + 1]; w = c.iloc[i + 1:i + 1 + W]
        return {"date": d, "exec": P, "fwd_min": w.min() / P - 1, "fwd_max": w.max() / P - 1, "ret_end": w.iloc[-1] / P - 1, "n": len(w)}
    s8 = rules["DD-8%"]
    x = s8.loc["2007-10-31":]; ok = stats(x[x].index[0])
    ev = events(s8.loc[start:], k, dd)
    cands = [stats(d) for d in ev.date if c.index.get_loc(d) + 1 + W < len(c)]
    cands = [e for e in cands if e["fwd_min"] > -0.10]
    bad = max(cands, key=lambda e: e["ret_end"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, e, kind in zip(axes, (ok, bad), ("성공", "실패(오경보)")):
        i = c.index.get_loc(e["date"])
        a, b = max(i - 120, 0), min(i + 1 + W, len(c) - 1)
        w = c.iloc[a:b]
        P = e["exec"]
        pk = c.iloc[max(i - 120, 0):i + 1].idxmax()
        ax.plot(w.index, w, color="#1f3b73", lw=1.6)
        after = w.loc[c.index[i + 1]:]
        ax.hlines(P, c.index[i + 1], w.index[-1], color="k", ls="--", lw=1)
        ax.fill_between(after.index, after, P, where=(after < P).values, color=GREEN, alpha=0.35, interpolate=True, label="팔고 나서 더 떨어진 만큼 = 아낀 손실")
        ax.fill_between(after.index, after, P, where=(after > P).values, color=RED, alpha=0.30, interpolate=True, label="팔고 나서 오른 만큼 = 놓친 수익")
        ax.scatter([pk], [c.loc[pk]], color=ORANGE, s=120, zorder=5); ax.annotate("직전 고점", (pk, c.loc[pk]), textcoords="offset points", xytext=(-10, 12), color=ORANGE, fontweight="bold")
        ax.scatter([c.index[i + 1]], [P], color=RED, marker="v", s=200, zorder=6)
        ax.annotate(f"여기서 전량 매도\n(고점 대비 {dd.iloc[i]:.0%} 하락한 다음날)", (c.index[i + 1], P), textcoords="offset points", xytext=(30, -60 if kind == "성공" else 60), arrowprops=dict(arrowstyle="->"), fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))
        ax.set_title(f"{kind}: {e['date'].year}년 {e['date'].month}월 발동")
        ax.text(0.02 if kind == "성공" else 0.98, 0.04, f"매도 후 1년(250거래일):\n 가장 많이 떨어졌을 때 {e['fwd_min']:+.0%}\n 가장 많이 올랐을 때 {e['fwd_max']:+.0%}\n 1년 뒤 {e['ret_end']:+.0%}",
                transform=ax.transAxes, va="bottom", ha="left" if kind == "성공" else "right", bbox=dict(boxstyle="round", fc="white", ec=GRAY), fontsize=12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=11, ncol=1, frameon=False); ax.grid(alpha=0.25); ax.set_ylabel("KOSPI")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
    fig.suptitle("① 규칙이 하는 일: '고점 대비 -8% 빠지면 전량 매도'", fontsize=18, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "explain_1_규칙예시.png", dpi=110); plt.close(fig)


def fig2():
    r = pd.read_csv(OUT / "sell_lab_rules.csv", index_col=0)
    names = {"기본 DD-8%": "기본\n(-8%면\n바로 매도)", "DD-8% & 수급≥70": "+수급\n확인", "DD-8% & 민감도≥70": "+시장민감도\n확인",
             "DD-8% & 심리≥70": "+심리\n확인", "DD-8% & 3그룹 중 2개≥60": "+세 그룹 중\n2개 확인", "DD-8% & 하나라도≥70": "+하나라도\n확인"}
    r = r.loc[list(names)]
    cols = [RED] + [BLUE] * 5
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    specs = [("평균 회피비율", "큰 하락에서 손실을 평균 몇 % 피했나\n(높을수록 좋음)", lambda v: f"{v:.0%}", 1.0),
             ("발동 지연(일,중앙)", "고점 이후 매도까지 걸린 날 수\n(짧을수록 빠름)", lambda v: f"{v:.0f}일", None),
             ("유효율", "매도 뒤 정말 -10% 더 떨어진 비율\n(높을수록 오경보 적음)", lambda v: f"{v:.0%}", 1.0)]
    for a, (col, ttl, fmt, ymax) in zip(ax, specs):
        v = r[col].values
        bars = a.bar(range(len(v)), v, color=cols)
        for x, val in enumerate(v):
            a.text(x, val, fmt(val), ha="center", va="bottom", fontweight="bold")
        a.set_xticks(range(len(v))); a.set_xticklabels([names[i] for i in r.index], fontsize=10.5)
        a.set_title(ttl, fontsize=14)
        if ymax: a.set_ylim(0, ymax)
        a.grid(alpha=0.25, axis="y")
    fig.suptitle("② 확인 조건을 더 붙이면? → 매도가 늦어져서 덜 피하게 됨 (오경보 감소는 미미)", fontsize=18, fontweight="bold")
    fig.text(0.5, -0.02, "빨강 = 기본 규칙, 파랑 = 수급·시장민감도·심리 지표로 한 번 더 확인한 규칙 (2006~2026, 큰 하락 8번)", ha="center", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "explain_2_규칙비교.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def fig3():
    e = pd.read_csv(OUT / "sell_lab_episodes.csv", index_col=0)
    d = pd.read_csv(OUT / "sell_lab_episode_dates.csv")
    names = {"기본 DD-8%": "기본 (-8%면 매도)", "DD-8% & 수급≥70": "+수급 확인", "DD-8% & 민감도≥70": "+시장민감도 확인", "DD-8% & 심리≥70": "+심리 확인",
             "DD-8% & 3그룹 중 2개≥60": "+세 그룹 중 2개 확인", "DD-8% & 하나라도≥70": "+하나라도 확인"}
    M = e[list(names)].T.values
    fig, ax = plt.subplots(figsize=(17, 6.5))
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "못 피함" if M[i, j] == 0 else f"{M[i, j]:.0%}", ha="center", va="center", fontsize=13,
                    color="#868e96" if M[i, j] == 0 else ("white" if M[i, j] > 0.6 else "black"), fontweight="bold")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{str(p)[:7]} 고점\n전체 {x:.0%} 하락" for p, x in zip(d["peak"], d["depth"])], fontsize=11)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(list(names.values()), fontsize=12)
    ax.set_title("③ 큰 하락 8번에서, 각 규칙이 전체 하락의 몇 %를 피했나 (진한 초록 = 많이 피함)", fontsize=16, fontweight="bold", pad=14)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "explain_3_구간별.png", dpi=110); plt.close(fig)


def fig4():
    t = pd.read_csv(OUT / "sell_lab_auc.csv").sort_values("AUC")
    fig, ax = plt.subplots(figsize=(15, 9))
    y = np.arange(len(t))
    colr = [GREEN if "신호O" in j else GRAY for j in t["판정"]]
    ax.barh(y, t["AUC"] - 0.5, left=0.5, color=colr, height=0.65)
    ax.errorbar(t["AUC"], y, xerr=[t["AUC"] - t["CI하한"], t["CI상한"] - t["AUC"]], fmt="none", ecolor="#495057", lw=1.2, capsize=3)
    ax.axvline(0.5, color="k", lw=1.5)
    ax.set_ylim(-1.6, len(t) - 0.4)
    ax.text(0.5, -1.1, "▲ 0.5 = 동전 던지기 수준", ha="center", fontweight="bold", fontsize=12, bbox=dict(fc="white", ec="none"))
    ax.set_yticks(y); ax.set_yticklabels([f"[{g}] {n}" for g, n in zip(t["그룹"], t["지표"])], fontsize=12)
    ax.set_xlim(0.25, 0.78)
    ax.set_xlabel("← 오히려 반대로 맞힘 | 0.5 | 이 지표가 높을수록 추가 하락이 많았음 →")
    ax.set_title("④ 어떤 지표가 '더 떨어질지'를 구분했나? 초록은 단 하나뿐", fontsize=15, fontweight="bold")
    ax.text(0.99, 0.02, "가는 선 = 오차 범위(95%).\n선이 0.5를 걸치면 우연일 수 있음", transform=ax.transAxes, ha="right", va="bottom", bbox=dict(boxstyle="round", fc="white", ec=GRAY), fontsize=12)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "explain_4_지표구분력.png", dpi=110); plt.close(fig)


def fig5(k, rules, dd, vix, spx):
    c = k["close"]
    pk = c.loc["2026-01-01":].idxmax()
    w = c.loc["2026-01-01":]
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.plot(w.index, w, color="#1f3b73", lw=2)
    ax.scatter([pk], [c.loc[pk]], color=ORANGE, s=150, zorder=5)
    ax.annotate(f"고점 {c.loc[pk]:,.0f}\n({pk.date()})", (pk, c.loc[pk]), textcoords="offset points", xytext=(-25, 6), ha="right", color=ORANGE, fontweight="bold", fontsize=13)
    show = [("DD-8%", "고점 -8%"), ("DD-10%", "고점 -10%"), ("DD-20%", "고점 -20%"), ("200일선 이탈(3일)", "200일선 이탈"), ("데드크로스(3일)", "데드크로스")]
    cols = [RED, "#d9480f", "#9c36b5", BLUE, GREEN]
    offs = [(85, 48), (85, 0), (85, -48), (-40, 110), (40, 70)]
    for j, ((n, lab), colr) in enumerate(zip(show, cols)):
        s = rules[n]; x = s.loc[pk:]
        f = x[x & ~s.shift(1, fill_value=False).loc[pk:]]
        if len(f):
            d = f.index[0]
            ax.scatter([d], [c.loc[d]], color=colr, marker="v", s=170, zorder=6)
            ax.annotate(f"{lab} 규칙 발동\n{d.strftime('%m/%d')} (그날 낙폭 {dd.loc[d]:.0%})", (d, c.loc[d]), textcoords="offset points", xytext=offs[j], ha="left" if j != 3 else "right", color=colr, fontweight="bold", fontsize=12,
                        arrowprops=dict(arrowstyle="-", color=colr), bbox=dict(boxstyle="round", fc="white", ec=colr, alpha=0.9))
    ax.scatter([c.index[-1]], [c.iloc[-1]], color="k", s=120, zorder=7)
    ax.annotate(f"현재 {c.iloc[-1]:,.0f}\n(고점 대비 {c.iloc[-1] / c.loc[pk] - 1:.0%})", (c.index[-1], c.iloc[-1]), textcoords="offset points", xytext=(-10, -62), ha="right", fontweight="bold", fontsize=13)
    ax.set_ylim(top=9900)
    ax.set_title("⑤ 지금은 어디쯤? 하락 시작을 알리는 규칙들은 이미 모두 울린 상태", fontsize=17, fontweight="bold")
    ax.grid(alpha=0.25); ax.set_ylabel("KOSPI")
    fig.tight_layout(); fig.savefig(OUT / "explain_5_현재.png", dpi=110); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    k = pd.read_csv(data.DATA_DIR / "KOSPI_ohlcv.csv", index_col=0, parse_dates=True)
    vix, spx = data.load_series("VIX"), data.load_series("SPX")
    rules, dd = build_rules(k, vix, spx)
    fig1(k, rules, dd); fig2(); fig3(); fig4(); fig5(k, rules, dd, vix, spx)
    print("완료:", sorted(p.name for p in OUT.glob("explain_*.png")))


if __name__ == "__main__":
    main()
