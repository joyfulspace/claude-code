"""KOSPI 큰 하락(전액 매도급) 조기경보 평가: python3 backtest/run_crash_warning.py

목표(label): '앞으로 H(60)거래일 안에 오늘 종가 대비 -15% 이하로 하락'  (보조: 20일 내 -10%)
1) 지표별 AUC(+블록 부트스트랩 CI, 전/후반 기간 방향 일관성)
2) 하락 구간(고점→저점 -15%↑)별로 경보가 고점 전/후 언제 울렸는지
3) 지표 결합 로지스틱 모델은 워크포워드(과거로만 학습 → 다음 해 예측)로 검증
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import risk_exit as rx  # noqa: E402

OUT = Path(__file__).parent / "results"
H, DROP = 60, -0.15


def auc(score: pd.Series, y: pd.Series) -> float:
    d = pd.concat([score, y], axis=1).dropna()
    if len(d) == 0 or d.iloc[:, 1].nunique() < 2:
        return np.nan
    r = d.iloc[:, 0].rank()
    n1 = int(d.iloc[:, 1].sum()); n0 = len(d) - n1
    return (r[d.iloc[:, 1] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def block_ci(score, y, block=120, reps=300, seed=0):
    d = pd.concat([score, y], axis=1).dropna().values
    n = len(d); rng = np.random.default_rng(seed); vals = []
    nb = int(np.ceil(n / block))
    for _ in range(reps):
        st = rng.integers(0, n - block, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        s_, y_ = pd.Series(d[idx, 0]), pd.Series(d[idx, 1])
        vals.append(auc(s_, y_))
    return np.nanpercentile(vals, [2.5, 97.5])


def fit_logit(X, y, lam=30.0, iters=25):
    Xb = np.c_[np.ones(len(X)), X]; w = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w)); g = Xb.T @ (p - y) + R @ w
        Hs = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(Hs, g)
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="2013-01-01"); a = ap.parse_args()
    pd.options.display.float_format = "{:.3f}".format
    OUT.mkdir(exist_ok=True)
    k = pd.read_csv(data.DATA_DIR / "KOSPI_ohlcv.csv", index_col=0, parse_dates=True)
    vix = data.load_series("VIX")
    funds = pd.read_csv(data.DATA_DIR / "KOFIA_funds.csv", index_col=0, parse_dates=True)
    c = k["close"]
    comp = rx.components(k, vix, funds)
    vix1 = vix.reindex(k.index).ffill().shift(1)
    comp["stress_drawdown"] = rx.pct_rank(-(c / c.rolling(252).max() - 1))
    comp["stress_vix_level"] = rx.pct_rank(vix1)
    comp["stress_realized_vol"] = rx.pct_rank(c.pct_change().rolling(20).std())

    y15 = (c.rolling(H).min().shift(-H) / c - 1 <= DROP).astype(float).where(c.rolling(H).min().shift(-H).notna())
    y10 = (c.rolling(20).min().shift(-20) / c - 1 <= -0.10).astype(float).where(c.rolling(20).min().shift(-20).notna())
    valid = comp.dropna().index
    print(f"표본 {valid[0].date()}~{c.index[-1].date()} | label: {H}일 내 {DROP:.0%} 하락 base rate {y15.loc[valid].mean():.1%} "
          f"(최근 {H}일은 라벨 미확정으로 제외)\n")

    # ---------- 1) 지표별 AUC
    rows = {}
    for col in comp.columns:
        s = comp[col]
        e = a.split
        a_all, a_in, a_out = auc(s, y15), auc(s.loc[:e], y15.loc[:e]), auc(s.loc[e:], y15.loc[e:])
        lo, hi = block_ci(s, y15)
        rows[col] = {"AUC": a_all, "CI하한": lo, "CI상한": hi, f"~{e[:4]}": a_in, f"{e[:4]}~": a_out,
                     "20일-10%": auc(s, y10),
                     "판정": ("신호O" if lo > 0.5 else "역방향" if hi < 0.5 else "불명확") + (" / 기간일관" if (a_in - .5) * (a_out - .5) > 0 else " / 기간불일치")}
    t1 = pd.DataFrame(rows).T
    t1 = t1.sort_values("AUC", ascending=False)
    print("[1] 지표별 하락 예측력 (AUC 0.5=무의미, >0.5 클수록 위험↑ 지표가 하락 선행, <0.5 반대 방향)")
    print(t1.to_string(), "\n")

    # ---------- 범주 점수
    cat = {}
    for name, prefix in (("국면", "regime_"), ("심리", "sent_"), ("거래량", "vol_"), ("수급", "supply_"), ("시장스트레스", "stress_")):
        cat[name] = comp[[x for x in comp.columns if x.startswith(prefix)]].mean(axis=1, skipna=False) * 100
    cat = pd.DataFrame(cat)

    # ---------- 3) 워크포워드 로지스틱 (모든 지표, 과거로만 학습)
    X = (comp - 0.5).fillna(0.0)
    ok = comp.dropna().index
    prob = pd.Series(np.nan, index=c.index)
    last_w = None
    for yr in range(2008, c.index[-1].year + 1):
        t0 = pd.Timestamp(f"{yr}-01-01"); t1_ = pd.Timestamp(f"{yr + 1}-01-01")
        tr = [d for d in ok if d < t0 - pd.Timedelta(days=int(H * 1.5)) and not np.isnan(y15.loc[d])]
        te = [d for d in c.index if t0 <= d < t1_ and d in comp.dropna().index]
        if len(tr) < 750 or not te:
            continue
        last_w = fit_logit(X.loc[tr].values, y15.loc[tr].values)
        prob.loc[te] = predict(last_w, X.loc[te].values)
    cat["모델(워크포워드)"] = prob
    oos = prob.dropna().index
    base = y15.loc[oos].mean()
    print(f"[2] 워크포워드 로지스틱(2008~ 연 단위 재학습) — OOS {oos[0].date()}~{oos[-1].date()}, base rate {base:.1%}")
    for nm, sl in (("OOS 전체", slice(None)), ("2008~2012", slice(None, "2012-12-31")), ("2013~2019", slice("2013-01-01", "2019-12-31")), ("2020~", slice("2020-01-01", None))):
        print(f"   {nm}: AUC {auc(prob.loc[sl], y15.loc[sl]):.3f}")
    coef = pd.Series(last_w[1:], index=comp.columns).sort_values()
    print("   최근 모델 계수(+ = 위험↑일 때 하락확률↑):", {k_: round(v, 2) for k_, v in coef.items()}, "\n")

    # ---------- 범주별 AUC & 경보 lift
    print("[3] 범주별 AUC / 경보(상위20%, 3일 연속) 시 하락 적중률")
    rows = {}
    for nm in cat.columns:
        s = cat[nm]
        if nm.startswith("모델"):
            thr = 2 * base; warn = (s >= thr).astype(int).rolling(3).sum() >= 3
            idx = oos
        else:
            warn = (s >= 80).astype(int).rolling(3).sum() >= 3
            idx = s.dropna().index
        w = warn.loc[idx]; yy = y15.loc[idx].dropna(); w = w.loc[yy.index]
        rows[nm] = {"AUC": auc(s, y15), f"~{a.split[:4]} AUC": auc(s.loc[:a.split], y15.loc[:a.split]), f"{a.split[:4]}~ AUC": auc(s.loc[a.split:], y15.loc[a.split:]),
                    "경보일 비중": w.mean(), "경보시 -15%↓확률": yy[w].mean() if w.any() else np.nan, "평소(base)": yy.mean()}
    print(pd.DataFrame(rows).T.to_string(), "\n")

    # ---------- 2) 하락 구간별 경보 시점
    ep = rx.drawdown_episodes(c.loc[comp.dropna().index[0]:], depth=DROP)
    print("[4] 큰 하락 구간별 경보 (고점 -120일 ~ -10% 도달일 사이 첫 경보; 선행일수=고점 기준 달력일, 음수=고점 이전)")
    erows = []
    for _, e in ep.iterrows():
        r = {"고점": e["peak"].date(), "낙폭": e["depth"]}
        lo_, hi_ = e["peak"] - pd.Timedelta(days=120), e["dd10_date"]
        for nm in ["수급", "시장스트레스", "국면", "심리", "거래량", "모델(워크포워드)"]:
            s = cat[nm]
            warn = (s >= (2 * base if nm.startswith("모델") else 80)).astype(int).rolling(3).sum() >= 3
            w = warn.loc[lo_:hi_]
            if nm.startswith("모델") and e["peak"] < oos[0]:
                r[nm] = "(학습구간)"; continue
            if w.any():
                d = w[w].index[0]
                r[nm] = f"{(d - e['peak']).days:+d}일({c.loc[d] / c.loc[:d].max() - 1:.0%})"
            else:
                r[nm] = "-"
        erows.append(r)
    print(pd.DataFrame(erows).to_string(index=False))
    print("   (괄호 = 경보 시점의 고점 대비 낙폭)\n")

    # ---------- 현재 상태
    cur = cat.dropna(subset=["수급"]).iloc[-1]
    print(f"[5] 현재 {cat.dropna(subset=['수급']).index[-1].date()} 범주 점수:", {k_: round(float(v), 1) for k_, v in cur.items() if not np.isnan(v)})
    p_now = prob.dropna().iloc[-1]
    bins = pd.qcut(prob.dropna(), 5, labels=False, duplicates="drop")
    emp = y15.loc[bins.index].groupby(bins).mean()
    qb = int(np.searchsorted(prob.dropna().quantile([.2, .4, .6, .8]).values, p_now))
    print(f"    모델 확률 {p_now:.1%} (OOS 평균 {base:.1%}) → 비슷한 확률 구간의 과거 실제 -15% 도달 빈도 {emp.iloc[min(qb, len(emp) - 1)]:.1%}")
    cur_dd = c.iloc[-1] / c.max() - 1
    print(f"    KOSPI 현재 고점 대비 {cur_dd:.1%}")

    # ---------- 차트
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = comp.dropna().index[0]
    fig, ax = plt.subplots(3, 1, figsize=(18, 12), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]})
    ax[0].semilogy(c.loc[st:].index, c.loc[st:], color="#1f3b73", lw=1)
    for _, e in ep.iterrows():
        for x in ax:
            x.axvspan(e["peak"], e["trough"], color="#e03131", alpha=0.12)
        ax[0].annotate(f"{e['depth']:.0%}", (e["peak"], c.loc[e["peak"]]), textcoords="offset points", xytext=(0, 8), color="#e03131", fontsize=9)
    ax[0].set_title("KOSPI (red = drawdown >= 15%, peak -> trough)")
    for nm, colr in (("수급", "#2b8a3e"), ("시장스트레스", "#1971c2"), ("심리", "#f08c00")):
        ax[1].plot(cat.index, cat[nm], lw=0.9, color=colr, label={"수급": "Supply", "시장스트레스": "Stress", "심리": "Sentiment"}[nm])
    ax[1].axhline(80, color="k", ls="--", lw=0.7); ax[1].legend(loc="upper left"); ax[1].set_ylim(0, 100)
    ax[1].set_title("Category risk percentile (dashed = 80)")
    ax[2].plot(prob.index, prob, color="#9c36b5", lw=1, label="Walk-forward P(-15% within 60d)")
    ax[2].axhline(base, color="gray", ls="--", lw=0.8, label=f"base rate {base:.0%}")
    ax[2].axhline(2 * base, color="#e03131", ls="--", lw=0.8, label="warning = 2x base"); ax[2].legend(loc="upper left")
    ax[2].set_title("Model probability (out-of-sample, retrained yearly)")
    for x in ax: x.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "crash_warning.png", dpi=110)
    pd.concat([c.rename("close"), cat, y15.rename("y15")], axis=1).to_csv(OUT / "crash_warning_signals.csv")
    print(f"\n차트: {OUT / 'crash_warning.png'}")


if __name__ == "__main__":
    main()
