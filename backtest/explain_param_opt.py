"""파라미터 최적화 결과 그림(한글): python3 backtest/explain_param_opt.py  (먼저 run_param_opt.py 실행)"""
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, GREEN, BLUE, GRAY, ORANGE = "#e03131", "#2b8a3e", "#1971c2", "#adb5bd", "#f08c00"
GC = {"수급": GREEN, "시장민감도": BLUE, "심리": ORANGE, "기준": RED}
D_GRID = [-5, -6, -8, -10, -12, -15]; TH_GRID = [20, 30, 40, 50, 60, 70, 80, 90]


def main():
    SM = pd.read_csv(OUT / "param_opt_summary.csv")
    N = pd.read_pickle(OUT / "param_opt_null.pkl")
    nf95, no95 = SM["전체 우연기준(95%)"].iloc[0], SM["검증 우연기준(95%)"].iloc[0]
    base = SM[SM["지표"] == "기준(지표 없음)"].iloc[0]
    X = SM[SM["지표"] != "기준(지표 없음)"].copy()

    # ---- 그림 1: 점수 정의 + 최적값 위치
    fig, ax = plt.subplots(figsize=(14, 9))
    r, p = np.meshgrid(np.linspace(0.01, 1, 200), np.linspace(0.01, 1, 200))
    f = 2 * r * p / (r + p)
    cs = ax.contourf(r, p, f, levels=np.linspace(0, 1, 11), cmap="Greens", alpha=0.35)
    cl = ax.contour(r, p, f, levels=[0.3, 0.4, 0.5, 0.6, 0.7], colors="k", linewidths=0.6, alpha=0.6)
    ax.clabel(cl, fmt=lambda v: f"점수 {v:.1f}", fontsize=10)
    for _, row in X.iterrows():
        ax.scatter(row["회피비율"], row["유효율"], s=130, color=GC[row["그룹"]], edgecolor="k", zorder=5)
        ax.annotate(row["지표"], (row["회피비율"], row["유효율"]), textcoords="offset points", xytext=(7, 4), fontsize=10)
    ax.scatter(base["회피비율"], base["유효율"], marker="*", s=420, color=RED, edgecolor="k", zorder=6)
    ax.annotate("기준: 지표 없이\n고점-10%만", (base["회피비율"], base["유효율"]), textcoords="offset points", xytext=(-100, -34), color=RED, fontweight="bold")
    ax.set_xlim(0.45, 0.85); ax.set_ylim(0.25, 0.62)
    ax.set_xlabel("큰 하락에서 전체 낙폭의 몇 %를 피했나 (→ 오른쪽일수록 좋음)")
    ax.set_ylabel("매도 뒤 정말 -10% 더 떨어진 비율 (↑ 위쪽일수록 헛발질이 적음)")
    ax.set_title("'가장 하락을 잘 피했다' = 오른쪽 위일수록. 점수 = 두 값의 조화평균 (각 지표의 최적값 위치)", fontsize=14)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "opt_1_점수정의와위치.png", dpi=110); plt.close(fig)

    # ---- 그림 2: 기본값 / 최적(전체) / 검증
    d = SM.sort_values("검증점수(교차)")
    y = np.arange(len(d)); h = 0.27
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.barh(y + h, d["기본값점수"], h, color=GRAY, label="① 기본값 (최적화 전)")
    ax.barh(y, d["최적점수(전체)"], h, color="#74c0fc", label="② 전체 기간으로 최적화 (과적합 포함)")
    ax.barh(y - h, d["검증점수(교차)"], h, color=[GC[g] for g in d["그룹"]], label="③ 검증: 한 구간에서 찾은 값을 다른 구간에 적용")
    ax.axvline(nf95, color="#1971c2", ls="--", lw=1.5); ax.text(nf95, len(d) - 0.3, " 우연 기준선(②)\n 가짜 지표도 이 정도는 나옴", color="#1971c2", fontsize=11, va="top")
    ax.axvline(no95, color="k", ls="--", lw=1.5); ax.text(no95, len(d) - 3.2, " 우연 기준선(③)", fontsize=11, va="top")
    ax.set_yticks(y); ax.set_yticklabels([f"{r['지표']}" + (f"  (창 {int(r['최적창'])}일, θ{int(r['최적θ'])})" if r["그룹"] != "기준" else "") for _, r in d.iterrows()], fontsize=11)
    ax.set_xlabel("점수 (높을수록 좋음)")
    ax.set_title("최적화한 값으로 지표 비교: ③ 막대(검증)가 검은 점선을 넘어야 '우연 이상'", fontsize=15)
    ax.legend(loc="lower right", fontsize=11); ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "opt_2_지표비교.png", dpi=110); plt.close(fig)

    # ---- 그림 3: 시뮬레이션 지형 (상위 3개 지표)
    top = X.sort_values("검증점수(교차)", ascending=False).head(3)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, (_, row) in zip(axes, top.iterrows()):
        pn, wins, dflt = N["S"][row["지표"]]
        wi, hi = wins.index(int(row["최적창"])), 0 if row["방향"] == "높을때" else 1
        M = N["scores"][row["지표"]][0, wi, hi]
        im = ax.imshow(M, cmap="YlGn", vmin=0.25, vmax=0.6, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10.5, color="k")
        di, ti = D_GRID.index(int(row["최적D"])), TH_GRID.index(int(row["최적θ"]))
        ax.add_patch(plt.Rectangle((ti - .5, di - .5), 1, 1, fill=False, ec=RED, lw=3))
        ax.set_xticks(range(len(TH_GRID))); ax.set_xticklabels([f"{t}" for t in TH_GRID]); ax.set_yticks(range(len(D_GRID))); ax.set_yticklabels([f"{d_}%" for d_ in D_GRID])
        ax.set_xlabel("지표 임계 백분위 θ"); ax.set_ylabel("고점 대비 하락 D")
        ax.set_title(f"{row['지표']}\n(창 {int(row['최적창'])}일, 지표가 {row['방향']})", fontsize=14)
    fig.suptitle("시뮬레이션 모습: 칸 하나 = 파라미터 한 조합의 점수 (빨간 테두리 = 선택된 값)", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "opt_3_시뮬레이션지형.png", dpi=110); plt.close(fig)

    # ---- 그림 4: 비용 관점
    d = SM.sort_values("120일순절약합")
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(range(len(d)), d["120일순절약합"] * 100, color=[GC[g] for g in d["그룹"]])
    ax.axvline(0, color="k", lw=1.2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["지표"], fontsize=11)
    ax.set_xlabel("신호 때마다 팔고 120일 뒤 복귀했을 때, 그냥 보유한 것 대비 이득의 합 (%p, 0보다 오른쪽이어야 이득)")
    ax.set_title("점수가 높아도 실제로 '돈이 되는가'는 다른 문제: 모두 0보다 왼쪽(손해)", fontsize=15)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "opt_4_비용관점.png", dpi=110); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
