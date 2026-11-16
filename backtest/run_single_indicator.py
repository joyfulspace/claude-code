"""지표를 종합하지 않고 하나씩 확인 조건으로 붙여 평가: python3 backtest/run_single_indicator.py [--thr 70]
규칙 = (고점 대비 -8% 이하) AND (해당 지표 백분위 >= thr).  기본 규칙(-8%만)과 같은 방식으로 비교."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import risk_exit as rx  # noqa: E402
from run_decline_response import events, GOOD  # noqa: E402
from run_sell_lab import indicators, load_all  # noqa: E402

OUT = Path(__file__).parent / "results"


def evaluate(cond, k, dd, start, base_ep):
    c, o = k["close"], k["open"]
    ev = events(cond.fillna(False).loc[start:], k, dd)
    done = ev[ev.complete] if len(ev) else ev
    good = (done.fwd_min <= GOOD) if len(done) else pd.Series(dtype=bool)
    av, dl = [], []
    for _, ep in base_ep.iterrows():
        w = cond.fillna(False).loc[ep["peak"]:ep["trough"]]
        if not w.any():
            av.append(0.0); continue
        d = w[w].index[0]; i = c.index.get_loc(d)
        P = o.iloc[i + 1] if i + 1 < len(c) else c.iloc[i]
        av.append(max((P - c.loc[ep["trough"]]) / (c.loc[ep["peak"]] - c.loc[ep["trough"]]), 0)); dl.append((d - ep["peak"]).days)
    return {"발동": len(ev), "유효율": good.mean() if len(done) else np.nan, "오경보": int((~good).sum()) if len(done) else 0,
            "평균회피": float(np.mean(av)), "회피25%↑": int(sum(x >= .25 for x in av)), "지연(일)": float(np.median(dl)) if dl else np.nan}, av


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, default=70); a = ap.parse_args()
    pd.options.display.float_format = "{:.2f}".format
    k, ser, funds = load_all()
    c = k["close"]
    ind, _ = indicators(k, ser, funds)
    dd = c / c.rolling(252, min_periods=60).max() - 1
    dd8 = dd <= -0.08
    start = ind.dropna().index[0]
    base_ep = rx.drawdown_episodes(c.loc[start:], depth=-0.15)
    base, base_av = evaluate(dd8, k, dd, start, base_ep)
    res, avs = {("기준", "고점 -8%만"): base}, {("기준", "고점 -8%만"): base_av}
    for key in ind.columns:
        res[key], avs[key] = evaluate(dd8 & (ind[key] >= a.thr / 100), k, dd, start, base_ep)
    R = pd.DataFrame(res).T
    R.index = [f"[{g}] {n}" for g, n in R.index]
    print(f"평가 구간 {start.date()}~ (큰 하락 {len(base_ep)}개), 규칙 = 고점 -8% & 지표 백분위 ≥ {a.thr:.0f}\n")
    print(R.sort_values("평균회피", ascending=False).to_string(), "\n")
    # 임계 민감도
    sens = {}
    for thr in (50, 60, 70, 80):
        sens[f"≥{thr}"] = {f"[{g}] {n}": evaluate(dd8 & (ind[(g, n)] >= thr / 100), k, dd, start, base_ep)[0]["평균회피"] for g, n in ind.columns}
    print("[임계 민감도] 평균 회피비율 (기준 %.2f)" % base["평균회피"]); print(pd.DataFrame(sens).to_string(), "\n")
    R.to_csv(OUT / "single_indicator_rules.csv", encoding="utf-8-sig")
    E = pd.DataFrame(avs, index=[str(p.date()) for p in base_ep["peak"]]).T
    E.index = [f"[{g}] {n}" for g, n in E.index]; E.to_csv(OUT / "single_indicator_episodes.csv", encoding="utf-8-sig")
    base_ep.to_csv(OUT / "single_indicator_episode_dates.csv", index=False)

    # ---- 그림
    import koreanize_matplotlib  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    cmap = {"수급": "#2b8a3e", "시장민감도": "#1971c2", "심리": "#f08c00"}
    fig, ax = plt.subplots(figsize=(15, 9))
    for lab, row in R.iloc[1:].iterrows():
        g = lab[1:lab.index("]")]; nm = lab[lab.index("]") + 2:]
        ax.scatter(row["지연(일)"], row["평균회피"], s=60 + row["발동"] * 12, color=cmap[g], alpha=0.8, edgecolor="k", lw=0.5)
        dx, dy, ha = {"미국 베타(60일)": (-9, 5, "right"), "반대매매 비중": (9, 6, "left"), "미국금리 상승(60일)": (-9, -14, "right"), "한미 상관(60일)": (9, 6, "left"),
                      "VIX 수준": (-9, 5, "right"), "예탁금 감소(60일)": (9, -14, "left"), "VIX 변화 민감도": (-9, 2, "right")}.get(nm, (7, 5, "left"))
        ax.annotate(nm, (row["지연(일)"], row["평균회피"]), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=10.5)
    ax.scatter(base["지연(일)"], base["평균회피"], s=420, marker="*", color="#e03131", zorder=6)
    ax.annotate("기준: 고점 -8%만\n(확인 조건 없음)", (base["지연(일)"], base["평균회피"]), textcoords="offset points", xytext=(22, -2), color="#e03131", fontweight="bold", fontsize=13)
    ax.set_xlabel("고점 이후 매도까지 걸린 날 수 (왼쪽일수록 빠름)"); ax.set_ylabel("큰 하락에서 평균 몇 %를 피했나 (위쪽일수록 좋음)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"지표를 하나씩 확인 조건으로 붙이면? (고점 -8% & 지표 상위 {100 - a.thr:.0f}%) — 별(기준)보다 위·왼쪽으로 간 지표가 없음", fontsize=14, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=v, label=k_, markersize=10) for k_, v in cmap.items()], loc="upper right", title="지표 그룹 (점 크기=발동 횟수)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "single_1_지표별.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 9))
    order = R.iloc[1:].sort_values("평균회피", ascending=False).index.tolist()
    rows = ["[기준] 고점 -8%만"] + order
    M = E.loc[rows].values
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "·" if M[i, j] == 0 else f"{M[i, j]:.0%}", ha="center", va="center", fontsize=10.5, color="#adb5bd" if M[i, j] == 0 else ("white" if M[i, j] > .6 else "black"))
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{p[:7]}\n({d:.0%})" for p, d in zip(E.columns, base_ep["depth"])], fontsize=11)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(rows, fontsize=11)
    ax.set_title("큰 하락 8번에서 지표별 확인 조건이 피한 비율 (진한 초록=많이 피함, ·=못 피함)", fontsize=15, fontweight="bold", pad=12)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "single_2_구간별.png", dpi=110); plt.close(fig)
    print("그림: single_1_지표별.png, single_2_구간별.png")


if __name__ == "__main__":
    main()
