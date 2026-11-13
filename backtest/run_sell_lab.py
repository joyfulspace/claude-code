"""매도 신호 실험실: 하락 시작(고점 대비 -8%) 시점에 수급·시장민감도·심리 지표가 '이후 추가 하락'을 구분하는지 검증하고,
확인 조건을 붙인 매도 규칙을 기본 규칙과 비교한다.   python3 backtest/run_sell_lab.py

지표 방향은 이론으로 미리 고정(높을수록 위험). 데이터를 보고 뒤집지 않고, 방향이 반대로 나오면 AUC<0.5 로 그대로 보여준다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import risk_exit as rx  # noqa: E402
from run_crash_warning import auc, block_ci  # noqa: E402
from run_decline_response import events, FWD, GOOD  # noqa: E402

OUT = Path(__file__).parent / "results"
THR = 70  # 그룹 확인 임계(상위 30%), 사전 고정


def load_all():
    k = pd.read_csv(data.DATA_DIR / "KOSPI_ohlcv.csv", index_col=0, parse_dates=True)
    ser = {n: data.load_series(n) for n in ("VIX", "SPX", "USDKRW", "TNX", "KOSDAQ")}
    funds = pd.read_csv(data.DATA_DIR / "KOFIA_funds.csv", index_col=0, parse_dates=True)
    return k, ser, funds


def indicators(k, ser, funds):
    c, idx = k["close"], k.index
    ret = c.pct_change()
    lag = lambda s: s.reindex(idx).ffill().shift(1)  # 한국 마감 시점에 확정된 값만
    spx, vix, fx, tnx = lag(ser["SPX"]), lag(ser["VIX"]), lag(ser["USDKRW"]), lag(ser["TNX"])
    kq = ser["KOSDAQ"].reindex(idx).ffill()
    us, fxr, vixc = spx.pct_change(), fx.pct_change(), vix.pct_change()
    f = funds.reindex(idx).ffill().shift(1)
    P = rx.pct_rank
    down, up = us < 0, us > 0
    beta = lambda m: ((ret * us).where(m).rolling(120, min_periods=30).mean() / (us ** 2).where(m).rolling(120, min_periods=30).mean())
    g = {}
    g["수급"] = {
        "신용융자/예탁금": P(f["margin"] / f["deposit"]),
        "예탁금 감소(60일)": 1 - P(f["deposit"].pct_change(60)),
        "반대매매 비중": P(f["forced_ratio"].rolling(20).mean()),
        "신용융자 증가(60일)": P(f["margin"].pct_change(60)),
        "원화 약세(60일)": P(fx.pct_change(60)),
    }
    g["시장민감도"] = {
        "한미 상관(60일)": P(ret.rolling(60).corr(us)),
        "미국 베타(60일)": P(ret.rolling(60).cov(us) / us.rolling(60).var()),
        "하락비대칭 베타(하락-상승)": P(beta(down) - beta(up)),
        "VIX 변화 민감도": P(-ret.rolling(60).corr(vixc)),
        "환율 민감도(원화약세시 하락)": P(-ret.rolling(60).corr(fxr)),
        "미국금리 상승(60일)": P(tnx.diff(60)),
    }
    g["심리"] = {
        "VIX 수준": P(vix),
        "VIX 급등(5일)": P(vix.pct_change(5)),
        "코스닥 상대약세(20일)": P(-(kq / c).pct_change(20)),
        "하락일 비중(20일)": P((ret < 0).astype(float).rolling(20).mean()),
        "RSI 과매도": 1 - P(rx.rsi(c)),
        "RSI 과열": P(rx.rsi(c)),
    }
    flat = {(grp, n): s for grp, d in g.items() for n, s in d.items()}
    ind = pd.DataFrame(flat)
    comp = pd.DataFrame({grp: pd.concat(list(d.values()), axis=1).mean(axis=1, skipna=False) * 100 for grp, d in g.items()})
    return ind, comp


def main():
    pd.options.display.float_format = "{:.3f}".format
    OUT.mkdir(exist_ok=True)
    k, ser, funds = load_all()
    c, o = k["close"], k["open"]
    ind, comp = indicators(k, ser, funds)
    dd = c / c.rolling(252, min_periods=60).max() - 1
    fmin60 = c.rolling(60).min().shift(-60) / c - 1
    y = (fmin60 <= GOOD).astype(float).where(fmin60.notna())

    # ---------- 1) 하락 상태(고점 대비 -8% 이하) 일별 표본에서 지표별 AUC
    state = (dd <= -0.08) & ind.notna().all(axis=1) & y.notna()
    n_state = int(state.sum())
    print(f"[1] '하락 상태'(고점 대비 -8% 이하) 일 {n_state}일, 이후 60일 내 -10% 추가 하락 비율 {y[state].mean():.1%}")
    print("    AUC > 0.5: 이 지표가 높을수록 추가 하락↑(이론 방향 적중), < 0.5: 반대. CI는 120일 블록 부트스트랩(표본이 겹치므로 참고용)")
    rows = []
    for key in ind.columns:
        s = ind[key]
        lo, hi = block_ci(s[state], y[state], reps=200)
        h1, h2 = auc(s[state].loc[:"2012"], y[state].loc[:"2012"]), auc(s[state].loc["2013":], y[state].loc["2013":])
        rows.append({"그룹": key[0], "지표": key[1], "AUC": auc(s[state], y[state]), "CI하한": lo, "CI상한": hi, "~2012": h1, "2013~": h2,
                     "판정": ("신호O" if lo > .5 else "역방향" if hi < .5 else "불명확") + ("·일관" if (h1 - .5) * (h2 - .5) > 0 else "·불일치")})
    t1 = pd.DataFrame(rows)
    print(t1.to_string(index=False), "\n")

    # ---------- 2) 그룹 점수 AUC
    print("[2] 그룹 점수(이론 방향 지표의 평균 백분위) AUC")
    rows = {}
    for grp in comp.columns:
        s = comp[grp]
        lo, hi = block_ci(s[state], y[state], reps=200)
        rows[grp] = {"AUC": auc(s[state], y[state]), "CI하한": lo, "CI상한": hi,
                     "~2012": auc(s[state].loc[:"2012"], y[state].loc[:"2012"]), "2013~": auc(s[state].loc["2013":], y[state].loc["2013":])}
    print(pd.DataFrame(rows).T.to_string(), "\n")

    # ---------- 3) 매도 규칙 비교
    dd8 = dd <= -0.08
    cond = {
        "기본 DD-8%": dd8,
        "DD-8% & 수급≥70": dd8 & (comp["수급"] >= THR),
        "DD-8% & 민감도≥70": dd8 & (comp["시장민감도"] >= THR),
        "DD-8% & 심리≥70": dd8 & (comp["심리"] >= THR),
        "DD-8% & 3그룹 중 2개≥60": dd8 & ((comp >= 60).sum(axis=1) >= 2),
        "DD-8% & 하나라도≥70": dd8 & ((comp >= THR).any(axis=1)),
    }
    start = comp.dropna().index[0]
    base_ep = rx.drawdown_episodes(c.loc[start:], depth=-0.15)
    ev = {n: events(s.fillna(False).loc[start:], k, dd) for n, s in cond.items()}
    rows, avoid, delays = {}, {}, {}
    for n, s in cond.items():
        e = ev[n]; done = e[e.complete]; good = done.fwd_min <= GOOD
        av, dl = [], []
        for _, ep in base_ep.iterrows():
            pk, tr = ep["peak"], ep["trough"]
            w = s.fillna(False).loc[pk:tr]
            if not w.any():
                av.append(0.0); continue
            d = w[w].index[0]
            i = c.index.get_loc(d)
            P = o.iloc[i + 1] if i + 1 < len(c) else c.iloc[i]
            av.append(max((P - c.loc[tr]) / (c.loc[pk] - c.loc[tr]), 0)); dl.append((d - pk).days)
        rows[n] = {"발동": len(e), "평가가능": len(done), "유효율": good.mean() if len(done) else np.nan, "오경보": int((~good).sum()),
                   "120일후 평균수익": done.ret_end.mean(), "평균 회피비율": np.mean(av), "회피25%↑ 구간": f"{sum(x >= .25 for x in av)}/{len(av)}",
                   "발동 지연(일,중앙)": np.median(dl) if dl else np.nan}
        avoid[n] = av
    print(f"[3] 매도 규칙 비교 (평가 구간 {start.date()}~, 큰 하락 {len(base_ep)}개)")
    print(pd.DataFrame(rows).T.to_string(), "\n")
    ep_t = pd.DataFrame({n: [f"{x:.0%}" if x > 0 else "-" for x in v] for n, v in avoid.items()})
    ep_t.insert(0, "고점", [e.date() for e in base_ep["peak"]]); ep_t.insert(1, "낙폭", [f"{x:.0%}" for x in base_ep["depth"]])
    print("[4] 큰 하락 구간별 회피비율"); print(ep_t.to_string(index=False), "\n")

    # ---------- 5) 현재
    last = comp.dropna().index[-1]
    print(f"[5] 현재 {last.date()}: 고점 대비 {dd.iloc[-1]:.1%} | 그룹 점수", {g_: round(float(comp.loc[last, g_]), 1) for g_ in comp.columns})
    cur = ind.loc[last].sort_values(ascending=False)
    print("    지표 상위(위험↑) 5:", [(a[1], round(float(v) * 100)) for a, v in cur.head(5).items()])
    print("    지표 하위(위험↓) 5:", [(a[1], round(float(v) * 100)) for a, v in cur.tail(5).items()])

    # ---------- 차트
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EN = {"수급": "Supply", "시장민감도": "Sensitivity", "심리": "Sentiment"}
    fig, ax = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [3, 2, 3]})
    cc = c.loc[start:]
    ax[0].semilogy(cc.index, cc, color="#1f3b73", lw=1)
    for _, ep in base_ep.iterrows():
        ax[0].axvspan(ep["peak"], ep["trough"], color="#e03131", alpha=0.10)
    for n, colr, mk in (("기본 DD-8%", "#e03131", "v"), ("DD-8% & 3그룹 중 2개≥60", "#2b8a3e", "D")):
        e = ev[n]; good = (e.fwd_min <= GOOD) & e.complete; bad = (e.fwd_min > GOOD) & e.complete
        lab = "DD-8%" if n.startswith("기본") else "DD-8% & 2 of 3 groups>=60"
        ax[0].scatter(e.date[good], cc.loc[e.date[good]], marker=mk, color=colr, s=70, label=f"{lab} (valid)", zorder=5)
        ax[0].scatter(e.date[bad], cc.loc[e.date[bad]], marker=mk, facecolors="none", edgecolors=colr, s=70, label=f"{lab} (false alarm)", zorder=5)
    ax[0].legend(loc="upper left", fontsize=8, ncol=2); ax[0].grid(alpha=0.25); ax[0].set_title("KOSPI: trigger events (filled = further drop >=10% within 120d)")
    for grp, colr in zip(comp.columns, ("#2b8a3e", "#1971c2", "#f08c00")):
        ax[1].plot(comp.index, comp[grp], lw=0.8, color=colr, label=EN[grp])
    ax[1].fill_between(dd.index, 0, 100, where=(dd <= -0.08).values, color="gray", alpha=0.12)
    ax[1].axhline(THR, color="k", ls="--", lw=0.7); ax[1].set_ylim(0, 100); ax[1].legend(loc="upper left", ncol=3)
    ax[1].set_title("Group risk percentile (gray = decline state, dashed = 70)"); ax[1].grid(alpha=0.25)
    EN_IND = {"신용융자/예탁금": "Margin/Deposit", "예탁금 감소(60일)": "Deposit drain", "반대매매 비중": "Forced sale", "신용융자 증가(60일)": "Margin growth", "원화 약세(60일)": "KRW weakness",
              "한미 상관(60일)": "US corr", "미국 베타(60일)": "US beta", "하락비대칭 베타(하락-상승)": "Downside asym beta", "VIX 변화 민감도": "VIX sensitivity",
              "환율 민감도(원화약세시 하락)": "FX sensitivity", "미국금리 상승(60일)": "UST yield rise", "VIX 수준": "VIX level", "VIX 급등(5일)": "VIX jump", "코스닥 상대약세(20일)": "KOSDAQ weak",
              "하락일 비중(20일)": "Down-day ratio", "RSI 과매도": "RSI oversold", "RSI 과열": "RSI overheated"}
    cmap = {"수급": "#2b8a3e", "시장민감도": "#1971c2", "심리": "#f08c00"}
    yy = np.arange(len(t1))
    ax[2].barh(yy, t1["AUC"] - 0.5, left=0.5, color=[cmap[g_] for g_ in t1["그룹"]])
    ax[2].errorbar(t1["AUC"], yy, xerr=[t1["AUC"] - t1["CI하한"], t1["CI상한"] - t1["AUC"]], fmt="none", ecolor="k", lw=0.8)
    ax[2].axvline(0.5, color="k", lw=0.8); ax[2].set_yticks(yy); ax[2].set_yticklabels([EN_IND[x] for x in t1["지표"]], fontsize=8); ax[2].invert_yaxis()
    ax[2].set_title("AUC in decline state (>0.5: higher indicator -> more further decline; whisker = 95% block-bootstrap CI)"); ax[2].grid(alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "sell_lab.png", dpi=110)
    t1.to_csv(OUT / "sell_lab_auc.csv", index=False)
    print(f"\n차트: {OUT / 'sell_lab.png'}")


if __name__ == "__main__":
    main()
