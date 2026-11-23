"""최적화 결과 그림(한글): python3 backtest/explain_avoid_opt.py  (먼저 run_avoid_opt.py)"""
import sys
import warnings
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from avoid_score import Scorer  # noqa: E402
from run_avoid_opt import Q, Ctx, sc  # noqa: E402
from run_param_opt import build_specs  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
GC = {"수급": "#2b8a3e", "시장민감도": "#1971c2", "심리": "#f08c00", "금리·환율": "#9c36b5"}
LAM = 0.1


def main():
    k, ser, funds = load_all(); S = Scorer(k); ctx = Ctx(S)
    R = pd.read_pickle(OUT / "avoid_opt_results.pkl"); A = pd.read_pickle(OUT / "avoid_opt_aux.pkl")
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl"); specs = build_specs(k, ser, funds)
    allE = np.ones(ctx.nE, bool)
    nl = A["null"][LAM]

    # ---- 그림 1: 지표 비교 (λ=0.1)
    T = R[R["λ"] == LAM].copy(); T["검증"] = (T["기간교차"] + T["구간별교차"]) / 2
    T = T.sort_values("검증")
    y = np.arange(len(T)); h = 0.27
    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.barh(y + h, T["기본값점수"], h, color="#adb5bd", label="① 기본값 (창 기본, 상위 20%)")
    ax.barh(y, T["최적점수(전체)"], h, color="#74c0fc", label="② 전체 기간으로 최적화 (과적합 포함)")
    ax.barh(y - h, T["검증"], h, color=[GC[g] for g in T["그룹"]], label="③ 검증: 기간 교차·구간별 교차 평균")
    ax.axvline(nl["full"][1], color="#1971c2", ls="--", lw=1.5); ax.text(nl["full"][1], len(T) - 0.4, " ② 우연 기준선(가짜 지표 95%)", color="#1971c2", fontsize=11, va="top")
    ax.axvline(nl["cv"][1], color="k", ls="--", lw=1.5); ax.text(nl["cv"][1], 0.2, " ③ 우연 기준선", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['지표']}  (창 {int(r['창'])}, 상위 {int(r['상위q%'])}%)" for _, r in T.iterrows()], fontsize=11)
    ax.set_xlabel("점수 (λ=0.1, 높을수록 좋음)"); ax.grid(alpha=0.25, axis="x"); ax.legend(loc="lower right", fontsize=11)
    ax.set_title("최적화한 값으로 지표 비교 (λ=0.1): ③ 막대가 점선을 넘어야 '우연 이상'")
    fig.tight_layout(); fig.savefig(OUT / "avo_1_지표비교.png", dpi=110); plt.close(fig)

    # ---- 그림 2: 신호 수가 같은 무작위와 비교
    rng = np.random.default_rng(5); n = S.n; rnd = []
    for _ in range(2500):
        cond = np.zeros(n, bool); cond[rng.choice(np.arange(S.start_i, n - 20), size=int(rng.integers(4, 120)), replace=False)] = True
        d = S.detail(cond)
        if len(d["ev"]): rnd.append((len(d["ev"]), sc(d, allE, np.ones(len(d["m"]), bool), LAM)))
    rnd = np.array(rnd)
    pts = []
    for (g, nm), (pn, wins, dflt, fn) in specs.items():
        for w in wins:
            for q in Q:
                d = S.detail(pct[(nm, w)] >= 1 - q / 100)
                if len(d["ev"]): pts.append((g, nm, len(d["ev"]), sc(d, allE, np.ones(len(d["m"]), bool), LAM)))
    P = pd.DataFrame(pts, columns=["g", "nm", "n", "s"])
    bins = np.arange(2, 46, 4); mid, lo, md, hi = [], [], [], []
    for a_, b_ in zip(bins[:-1], bins[1:]):
        v = rnd[(rnd[:, 0] >= a_) & (rnd[:, 0] < b_), 1]
        if len(v) > 20: mid.append((a_ + b_ - 1) / 2); lo.append(np.percentile(v, 5)); md.append(np.median(v)); hi.append(np.percentile(v, 95))
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.fill_between(mid, lo, hi, color="#ced4da", alpha=0.6, label="무작위 신호 (같은 신호 수에서 5~95% 범위)")
    ax.plot(mid, md, color="#868e96", lw=2, label="무작위 신호 중앙값")
    for g, c_ in GC.items():
        s_ = P[P["g"] == g]; ax.scatter(s_["n"], s_["s"], s=35, color=c_, alpha=0.75, label=f"{g} 지표의 파라미터 조합 하나하나")
    ax.set_xlabel("20년간 신호 횟수 (많을수록 자주 파는 규칙)"); ax.set_ylabel("점수 (λ=0.1)")
    ax.set_title("같은 신호 횟수의 무작위와 비교: 회색 띠를 넘는 점이 '지표 덕분'에 낫다고 볼 수 있는 규칙", fontsize=14)
    ax.legend(loc="lower left", fontsize=11); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "avo_2_무작위비교.png", dpi=110); plt.close(fig)
    # 몇 개가 띠를 넘는지
    def band95(nn):
        for a_, b_, h_ in zip(bins[:-1], bins[1:], hi):
            if a_ <= nn < b_: return h_
        return np.inf
    P["above"] = [s > band95(nn) for nn, s in zip(P["n"], P["s"])]
    print(f"무작위 95% 밴드를 넘은 조합 수(전체 {len(P)}개 중): {int(P['above'].sum())}개")
    print(P[P["above"]].groupby("nm").size().sort_values(ascending=False).to_string())
    P.to_csv(OUT / "avoid_opt_combos.csv", index=False, encoding="utf-8-sig")

    # ---- 그림 3: 큰 하락별 피한 손실 (λ=0.1 최적값)
    T = R[R["λ"] == LAM].copy(); T["검증"] = (T["기간교차"] + T["구간별교차"]) / 2
    top = T.sort_values("검증", ascending=False).head(6)
    M = np.array([r for r in top["B_j"]])
    fig, ax = plt.subplots(figsize=(16, 6.2))
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=0.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "못 피함" if M[i, j] == 0 else f"{M[i, j]:.0%}", ha="center", va="center", fontsize=12.5, color="#868e96" if M[i, j] == 0 else ("white" if M[i, j] > 0.3 else "black"), fontweight="bold")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{d[:7]} 고점\n전체 {x:.0%} 하락" for d, x in A["ep"]], fontsize=11)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"{r['지표']}\n(창 {int(r['창'])}, 상위 {int(r['상위q%'])}%, 신호 {int(r['신호'])}번)" for _, r in top.iterrows()], fontsize=11)
    ax.set_title("검증 상위 6개 지표: 큰 하락 8번에서 '피한 손실' (매도한 돈의 몇 %가 더 빠지는 걸 피했나)", fontsize=15, fontweight="bold", pad=12)
    for s_ in ax.spines.values(): s_.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "avo_3_구간별.png", dpi=110); plt.close(fig)
    # ---- 그림 4: λ에 따른 변화
    Rl = R.copy(); Rl["검증"] = (Rl["기간교차"] + Rl["구간별교차"]) / 2
    lams = sorted(Rl["λ"].unique())
    rank = Rl.pivot(index="지표", columns="λ", values="검증").rank(ascending=False).mean(axis=1).sort_values()
    sig = Rl.pivot(index="지표", columns="λ", values="신호")
    fig, ax = plt.subplots(1, 2, figsize=(18, 7.5))
    cols = plt.cm.tab10(np.arange(10))
    for i, nm in enumerate(rank.index[:6]):
        v = Rl[Rl["지표"] == nm].sort_values("λ")
        ax[0].plot(range(len(lams)), v["검증"], marker="o", lw=2.5, color=cols[i], label=nm)
        ax[1].plot(range(len(lams)), sig.loc[nm, lams], marker="o", lw=2.5, color=cols[i], label=nm)
    nul = [np.mean([A["null"][l]["cv"][1], A["null"][l]["loo"][1]]) for l in lams]
    ax[0].plot(range(len(lams)), nul, "k--", lw=2, label="우연 기준선 (가짜 지표 95%)")
    for a_ in ax:
        a_.set_xticks(range(len(lams))); a_.set_xticklabels([str(l) for l in lams]); a_.set_xlabel("λ (기회비용 가중치) → 오른쪽일수록 놓친 상승을 무겁게 봄"); a_.grid(alpha=0.25)
    ax[0].set_ylabel("검증 점수(기간교차·구간별교차 평균)"); ax[0].set_title("λ가 커질수록 점수는 줄고, 우연 기준선과의 차이가 사라져요"); ax[0].legend(fontsize=10.5)
    ax[1].set_ylabel("최적값에서의 신호 횟수 (20년)"); ax[1].set_title("λ가 클수록 최적값이 신호를 적게 내는 쪽으로 이동")
    fig.tight_layout(); fig.savefig(OUT / "avo_4_람다변화.png", dpi=110); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
