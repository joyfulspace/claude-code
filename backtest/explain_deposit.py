"""예탁금 감소 지표 상세 분석(한글): python3 backtest/explain_deposit.py  (먼저 run_avoid_opt.py)"""
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
from run_avoid_opt import Q, Ctx, sc  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
GREEN, RED, BLUE, ORANGE, GRAY = "#2b8a3e", "#e03131", "#1971c2", "#f08c00", "#868e96"
NM, W_OPT, Q_OPT = "예탁금 감소", 20, 5


def owners(S, ev):
    own = {}
    for i in ev:
        c = [(abs(i - pk), j) for j, (pk, tr, _, _) in enumerate(S.ep) if max(pk - PRE, S.start_i) <= i <= tr]
        if c: own[i] = min(c)[1]
    first = {}
    for i in ev:
        j = own.get(i)
        if j is not None and j not in first: first[j] = i
    return own, first


def main():
    k, ser, funds = load_all(); S = Scorer(k); ctx = Ctx(S); c = k["close"]; idx = k.index
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl"); R = pd.read_pickle(OUT / "avoid_opt_results.pkl"); A = pd.read_pickle(OUT / "avoid_opt_aux.pkl")
    f = funds.reindex(idx).ffill().shift(1)                              # 협회 자료는 다음 거래일 확정 -> 1일 지연
    dep = f["deposit"] / 1e6                                             # 조원
    raw = -dep.pct_change(W_OPT) * 100                                   # 예탁금 감소율(%) = -(N일 변화율)
    th95 = raw.expanding(min_periods=252).quantile(1 - Q_OPT / 100); th80 = raw.expanding(min_periods=252).quantile(0.8)
    p = pd.Series(pct[(NM, W_OPT)], index=idx)
    ev = S.events((p >= 1 - Q_OPT / 100).values); own, first = owners(S, ev)
    d = S.detail((p >= 1 - Q_OPT / 100).values)
    t0 = idx[S.start_i]
    ep_lab = [f"{str(e[2].date())[:7]} ({e[3]:.0%})" for e in S.ep]

    # ---- 신호표
    rows = []
    for i in ev:
        P = k["open"].iloc[i + 1]; w = c.iloc[i + 1:i + 121]
        j = own.get(i)
        rows.append({"신호일": idx[i].date(), "고점대비": c.iloc[i] / c.loc[:idx[i]].max() - 1, "예탁금(조원)": dep.iloc[i], "20일감소율(%)": raw.iloc[i],
                     "이후120일 최저(매도가대비)": w.min() / P - 1, "이후120일 최고": w.max() / P - 1, "120일 뒤": w.iloc[-1] / P - 1 if len(w) == 120 else np.nan,
                     "인정된 큰 하락": ep_lab[j] if j is not None else "-", "처음 인정": "★" if i in first.values() else ""})
    T = pd.DataFrame(rows); T.to_csv(OUT / "deposit_signals.csv", index=False, encoding="utf-8-sig")
    pd.options.display.float_format = "{:.3f}".format; pd.options.display.width = 250
    print(f"[{NM}] 창 {W_OPT}일, 상위 {Q_OPT}% → 신호 {len(ev)}번 (2006~), 큰 하락 중 {int((d['B'] > 0).sum())}/8 에서 인정")
    print(T.assign(고점대비=T["고점대비"].map("{:.0%}".format), **{"이후120일 최저(매도가대비)": T["이후120일 최저(매도가대비)"].map("{:+.0%}".format), "이후120일 최고": T["이후120일 최고"].map("{:+.0%}".format), "120일 뒤": T["120일 뒤"].map(lambda v: "-" if pd.isna(v) else f"{v:+.0%}")}).to_string(index=False))
    print("\n큰 하락별 인정 신호 / 피한 손실:")
    for j, (pk, tr, pkd, depth) in enumerate(S.ep):
        print(f"  {ep_lab[j]:<16} " + (f"{idx[first[j]].date()} (피한 손실 {d['B'][j]:.0%})" if j in first else "못 잡음"))
    done = T.dropna(subset=["120일 뒤"])
    base_dd = np.array([c.iloc[i + 1:i + 121].min() / k['open'].iloc[i + 1] - 1 for i in range(S.start_i, S.n - 121)])
    base_end = np.array([c.iloc[i + 120] / k['open'].iloc[i + 1] - 1 for i in range(S.start_i, S.n - 121)])
    print(f"\n신호 후 120일(완료 {len(done)}건): 평균 최저 {done['이후120일 최저(매도가대비)'].mean():+.1%}, 평균 최고 {done['이후120일 최고'].mean():+.1%}, 평균 120일 뒤 {done['120일 뒤'].mean():+.1%}, -10%↓ 비율 {(done['이후120일 최저(매도가대비)'] <= -0.1).mean():.0%}")
    print(f"아무 날이나(기준): 평균 최저 {base_dd.mean():+.1%}, 120일 뒤 평균 {base_end.mean():+.1%}, -10%↓ 비율 {(base_dd <= -0.1).mean():.0%}")
    last = p.dropna(); print(f"\n현재 {last.index[-1].date()}: 예탁금 {dep.iloc[-1]:.1f}조원, {W_OPT}일 감소율 {raw.iloc[-1]:+.1f}%, 백분위 {last.iloc[-1] * 100:.0f}%, 최근 신호 {[str(idx[i].date()) for i in ev[-3:]]}")

    # ---- 그림 1: 개요
    fig, ax = plt.subplots(4, 1, figsize=(19, 15), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.5, 1.7, 1.3]})
    cc = c.loc[t0:]; ax[0].semilogy(cc.index, cc, color="#1f3b73", lw=1.3)
    for pk, tr, _, depth in S.ep:
        for a in ax: a.axvspan(idx[pk], idx[tr], color=RED, alpha=0.10)
        ax[0].annotate(f"{depth:.0%}", (idx[pk], c.iloc[pk]), textcoords="offset points", xytext=(0, 8), color=RED, fontsize=10.5, fontweight="bold", ha="center")
    fi = set(first.values()); oin = [i for i in ev if i not in own]; ins = [i for i in ev if i in own and i not in fi]; cr = list(fi)
    if oin: ax[0].scatter(idx[oin], c.iloc[oin], marker="v", s=75, facecolors="white", edgecolors=GREEN, linewidths=1.6, zorder=4)
    if ins: ax[0].scatter(idx[ins], c.iloc[ins], marker="v", s=75, color=GREEN, alpha=0.65, zorder=4)
    if cr: ax[0].scatter(idx[cr], c.iloc[cr], marker="v", s=210, color=GREEN, edgecolors="k", linewidths=1.2, zorder=5)
    ax[0].set_title(f"예탁금 감소 지표 (창 {W_OPT}일, 상위 {Q_OPT}%): 신호 {len(ev)}번 — 큰 ▼=큰 하락마다 처음 인정, 작은 채움=큰 하락 구간 안, 빈 ▼=큰 하락과 무관"); ax[0].set_ylabel("KOSPI"); ax[0].grid(alpha=0.25)
    ax[1].plot(dep.loc[t0:].index, dep.loc[t0:], color=GREEN, lw=1.5); ax[1].set_title("투자자예탁금 (조원) — 원본", loc="left"); ax[1].grid(alpha=0.25)
    ax[2].plot(raw.loc[t0:].index, raw.loc[t0:], color=BLUE, lw=1.1, label=f"{W_OPT}거래일 감소율(%) = -(예탁금 {W_OPT}일 변화율)")
    ax[2].plot(th95.loc[t0:].index, th95.loc[t0:], "k--", lw=1.3, label=f"상위 {Q_OPT}% 경계 (오늘까지의 분포로 계산)"); ax[2].plot(th80.loc[t0:].index, th80.loc[t0:], color=GRAY, ls=":", lw=1.2, label="상위 20% 경계 (참고)")
    ax[2].scatter(idx[ev], raw.iloc[ev], marker="v", s=80, color=RED, edgecolors="k", zorder=5); ax[2].axhline(0, color="k", lw=0.6); ax[2].legend(fontsize=10.5, loc="upper left"); ax[2].set_title("예탁금 감소율(%) — 양수=예탁금이 줄어드는 중", loc="left"); ax[2].grid(alpha=0.25)
    ax[3].plot(p.loc[t0:].index, p.loc[t0:] * 100, color=ORANGE, lw=1.0); ax[3].axhline(100 - Q_OPT, color="k", ls="--", lw=1.2); ax[3].set_ylim(0, 100); ax[3].set_title("백분위(과거 대비 몇 % 수준인가) — 95 이상이면 신호", loc="left"); ax[3].grid(alpha=0.25)
    ax[3].xaxis.set_major_locator(mdates.YearLocator(2)); ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(OUT / "dep_1_개요.png", dpi=105); plt.close(fig)

    # ---- 그림 2: 신호별 결과 (이후 120일 최저/최고)
    fig, ax = plt.subplots(figsize=(19, 7.2)); xs = np.arange(len(T))
    ax.bar(xs, T["이후120일 최저(매도가대비)"] * 100, color=GREEN, alpha=0.8, label="이후 120일 최저 (매도가 대비, 피한 하락)")
    ax.bar(xs, T["이후120일 최고"] * 100, color=RED, alpha=0.6, label="이후 120일 최고 (매도가 대비, 놓친 상승)")
    ax.axhline(0, color="k", lw=0.8)
    for x_, (_, r) in zip(xs, T.iterrows()):
        if r["처음 인정"]: ax.text(x_, 3 + r["이후120일 최고"] * 100, "★", ha="center", fontsize=15, color="#e67700")
    ax.set_xticks(xs); ax.set_xticklabels([f"{str(d_)[:7]}\n({v:.0%})" for d_, v in zip(T["신호일"], T["고점대비"])], fontsize=9.5, rotation=0)
    ax.set_ylabel("매도가 대비 %"); ax.grid(alpha=0.25, axis="y"); ax.legend(loc="upper right", fontsize=11)
    ax.set_title("신호별 결과: 초록이 길수록 이후 더 빠졌고(잘 피함), 빨강이 길수록 이후 올랐음(놓친 상승). ★=큰 하락에서 처음 인정된 신호. 아래 괄호=신호일 고점 대비 낙폭", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "dep_2_신호별결과.png", dpi=105); plt.close(fig)

    # ---- 그림 3: 파라미터 지형 (창 x 상위%)
    wins = A["info"][NM][2]
    G = A["grids_raw"][NM]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    for a, lam in zip(axes[:2], (0.05, 0.1)):
        M = G[lam][0] if False else None
    nsig = np.array([[len(S.events(pct[(NM, w)] >= 1 - q / 100)) for q in Q] for w in wins])
    for a, (M, ttl, fmt, cm) in zip(axes, ((np.array([[sc(S.detail(pct[(NM, w)] >= 1 - q / 100), np.ones(ctx.nE, bool), np.ones(len(S.detail(pct[(NM, w)] >= 1 - q / 100)['m']), bool), 0.1) for q in Q] for w in wins]), "점수 (λ=0.1, 전체 기간)", "{:.2f}", "YlGn"),
                                  (nsig, "신호 횟수 (20년)", "{:.0f}", "Blues"),
                                  (np.array([[int((S.detail(pct[(NM, w)] >= 1 - q / 100)['B'] > 0).sum()) for q in Q] for w in wins]), "큰 하락 8번 중 잡은 수", "{:.0f}", "Reds"))):
        im = a.imshow(M, cmap=cm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): a.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")
        a.set_xticks(range(len(Q))); a.set_xticklabels([f"{q}%" for q in Q]); a.set_yticks(range(len(wins))); a.set_yticklabels([f"{w}일" for w in wins])
        a.set_xlabel("신호 기준: 감소율이 과거 대비 상위 몇 %일 때"); a.set_ylabel("예탁금 변화를 보는 기간"); a.set_title(ttl)
        a.add_patch(plt.Rectangle((Q.index(Q_OPT) - .5, wins.index(W_OPT) - .5), 1, 1, fill=False, ec="k", lw=3))
    fig.suptitle("파라미터 민감도 (검은 테두리 = 선택된 값: 20일, 상위 5%)", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "dep_3_파라미터.png", dpi=105); plt.close(fig)

    # ---- 그림 4: 검증 (λ별)
    r = R[R["지표"] == NM].sort_values("λ"); lams = list(r["λ"])
    fig, ax = plt.subplots(1, 2, figsize=(17, 6)); x = np.arange(len(lams)); wd = 0.26
    ax[0].bar(x - wd, r["최적점수(전체)"], wd, color="#74c0fc", label="전체 기간 최적화(낙관적)"); ax[0].bar(x, r["기간교차"], wd, color=GREEN, label="기간 교차검증"); ax[0].bar(x + wd, r["구간별교차"], wd, color=ORANGE, label="구간별 교차검증")
    for i, l in enumerate(lams):
        n = A["null"][l]
        for off, key in ((-wd, "full"), (0, "cv"), (wd, "loo")): ax[0].hlines(n[key][1], i + off - wd / 2, i + off + wd / 2, color="k", ls="--", lw=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"λ={l}\n(창 {int(w)}, 상위 {int(q)}%)" for l, w, q in zip(lams, r["창"], r["상위q%"])], fontsize=10.5)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].legend(fontsize=10.5); ax[0].set_title("점수와 우연 95% 기준선(검은 점선)"); ax[0].grid(alpha=0.25, axis="y")
    sig = r["신호"].values; ax[1].bar(x, sig, 0.5, color="#adb5bd", label="신호 횟수"); ax[1].bar(x, r["큰하락중신호"].values, 0.5, color=RED, label="큰 하락 8번 중 잡은 수")
    for i, (a_, b_) in enumerate(zip(sig, r["큰하락중신호"])): ax[1].text(i, a_, f"{int(a_)}", ha="center", va="bottom"); ax[1].text(i, b_, f"{int(b_)}", ha="center", va="bottom", color=RED)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"λ={l}" for l in lams]); ax[1].legend(fontsize=10.5); ax[1].set_title("λ가 클수록 최적값은 신호를 줄임"); ax[1].grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "dep_4_검증.png", dpi=105); plt.close(fig)

    # ---- 그림 5: 최근 확대
    a0, a1 = pd.Timestamp("2025-06-01"), idx[-1]; mm = (idx >= a0)
    fig, ax = plt.subplots(3, 1, figsize=(17, 9.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4, 1.6]})
    ax[0].plot(idx[mm], c[mm], color="#1f3b73", lw=1.6); ax[0].set_ylabel("KOSPI")
    for pk, tr, _, depth in S.ep:
        if idx[tr] >= a0:
            for a in ax: a.axvspan(max(idx[pk], a0), idx[tr], color=RED, alpha=0.10)
    evr = [i for i in ev if idx[i] >= a0]; ax[0].scatter(idx[evr], c.iloc[evr], marker="v", s=200, color=GREEN, edgecolors="k", zorder=5)
    for i in evr: ax[0].annotate(f"{idx[i].date()}\n(고점대비 {c.iloc[i] / c.loc[:idx[i]].max() - 1:.0%})", (idx[i], c.iloc[i]), textcoords="offset points", xytext=(10, 14), fontsize=10.5, fontweight="bold", color=GREEN)
    ax[1].plot(dep.loc[a0:].index, dep.loc[a0:], color=GREEN, lw=1.8); ax[1].set_title("투자자예탁금 (조원)", loc="left")
    ax[2].plot(raw.loc[a0:].index, raw.loc[a0:], color=BLUE, lw=1.5); ax[2].plot(th95.loc[a0:].index, th95.loc[a0:], "k--", lw=1.3, label=f"상위 {Q_OPT}% 경계"); ax[2].axhline(0, color="k", lw=0.6)
    ax[2].scatter(idx[evr], raw.iloc[evr], marker="v", s=120, color=RED, edgecolors="k", zorder=5); ax[2].legend(loc="upper left"); ax[2].set_title(f"{W_OPT}거래일 예탁금 감소율 (%)", loc="left")
    for a in ax: a.grid(alpha=0.25)
    ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
    fig.suptitle("최근 상황 (2025-06~)", fontsize=15, fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "dep_5_최근.png", dpi=105); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
