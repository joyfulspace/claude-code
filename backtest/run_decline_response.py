"""하락 시작 후 대응(전액 매도) 규칙 평가: python3 backtest/run_decline_response.py

- 신호는 종가 기준, 체결은 다음날 시가. 발동 이벤트 = 조건이 새로 참이 된 날(상승 에지), 이벤트 간 최소 120거래일.
- 유효: 발동 후 120거래일 내 체결가 대비 -10% 이하로 추가 하락. (아니면 오경보: 빠져나온 뒤 별 하락 없음)
- 회피비율(큰 하락 구간별): (체결가 - 저점) / (고점 - 저점). 100%=고점에서 팔기, 0%=저점에서 팔기, 음수=저점보다 낮게 팖(불가), 발동 없음='-'.
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
FWD, COOL, GOOD = 120, 120, -0.10


def confirm(cond: pd.Series, n: int = 3) -> pd.Series:
    return cond.fillna(False).astype(int).rolling(n).min().astype(bool)


def build_rules(k, vix, spx):
    c = k["close"]
    dd = c / c.rolling(252, min_periods=60).max() - 1
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    vix_pct = rx.pct_rank(vix.reindex(k.index).ffill().shift(1))
    vol_pct = rx.pct_rank(c.pct_change().rolling(20).std())
    us = spx.reindex(k.index).ffill().shift(1)
    us_dd = us / us.rolling(252, min_periods=60).max() - 1
    r = {
        "DD-8%": dd <= -0.08, "DD-10%": dd <= -0.10, "DD-15%": dd <= -0.15, "DD-20%": dd <= -0.20,
        "200일선 이탈(3일)": confirm(c < ma200),
        "데드크로스(3일)": confirm(ma50 < ma200),
        "변동성↑ & 50일선 이탈": (vol_pct >= 0.8) & (c < ma50),
        "DD-10% & VIX상위20%": (dd <= -0.10) & (vix_pct >= 0.8),
        "DD-10% & 200일선 아래": (dd <= -0.10) & (c < ma200),
        "20일 -10% 급락": c.pct_change(20) <= -0.10,
        "한미 동반 DD-10%": (dd <= -0.10) & (us_dd <= -0.10),
    }
    return {n: s.fillna(False) for n, s in r.items()}, dd


def events(cond: pd.Series, k: pd.DataFrame, dd: pd.Series):
    c, o = k["close"], k["open"]
    idx, out, last = cond.index, [], -10**9
    for i in range(1, len(idx) - 1):
        if cond.iloc[i] and not cond.iloc[i - 1] and i - last >= COOL:
            last = i
            P = o.iloc[i + 1]
            row = {"date": idx[i], "exec": P, "dd_at_trigger": dd.iloc[i], "complete": i + 1 + FWD < len(idx)}
            w = c.iloc[i + 1:i + 1 + FWD]
            row["fwd_min"], row["fwd_max"] = w.min() / P - 1, w.max() / P - 1
            row["ret_end"] = c.iloc[i + FWD] / P - 1 if row["complete"] else np.nan
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); a = ap.parse_args()
    pd.options.display.float_format = "{:.3f}".format
    OUT.mkdir(exist_ok=True)
    k = pd.read_csv(data.DATA_DIR / "KOSPI_ohlcv.csv", index_col=0, parse_dates=True)
    vix, spx = data.load_series("VIX"), data.load_series("SPX")
    rules, dd = build_rules(k, vix, spx)
    start = k.index[300]
    c = k["close"]
    ev = {n: events(s.loc[start:], k, dd) for n, s in rules.items()}

    print(f"평가 구간 {start.date()} ~ {c.index[-1].date()} | 유효 = 발동 후 {FWD}거래일 내 체결가 대비 {GOOD:.0%} 이하 추가 하락\n")
    rows = {}
    for n, e in ev.items():
        done = e[e.complete]
        good = (done.fwd_min <= GOOD)
        rows[n] = {"발동 횟수": len(e), "평가 가능": len(done), "유효율": good.mean() if len(done) else np.nan,
                   "오경보": int((~good).sum()), "평균 추가하락": done.fwd_min.mean(), "120일후 평균수익": done.ret_end.mean(), "평균 놓친반등(120일내 최고)": done.fwd_max.mean(),
                   "발동시 낙폭(중앙)": e.dd_at_trigger.median()}
    allv = pd.Series([c.iloc[i + FWD] / k["open"].iloc[i + 1] - 1 for i in range(len(c) - FWD - 1) if c.index[i] >= start], dtype=float)
    print(f"[참고] 아무 날이나 체결했을 때 120일 후 평균수익(무조건부) {allv.mean():.3f} — 규칙 발동 후 수익이 이보다 낮아야 빠져나온 보람이 있음")
    print("[1] 규칙별 발동 이벤트 평가")
    print(pd.DataFrame(rows).T.to_string(), "\n")

    # ---- 큰 하락 구간별 회피비율
    base = rx.drawdown_episodes(c.loc[start:], depth=-0.15)
    erows, avoid = [], {n: [] for n in rules}
    for _, e in base.iterrows():
        pk, tr = e["peak"], e["trough"]
        total = c.loc[pk] - c.loc[tr]
        r = {"고점": pk.date(), "저점": tr.date(), "낙폭": e["depth"]}
        for n, s in rules.items():
            w = s.loc[pk:tr]
            w = w[w & ~s.shift(1, fill_value=False).loc[pk:tr]] if w.any() else w
            if len(w) == 0 or not w.any():
                r[n] = "-"; avoid[n].append(0.0); continue
            d = w.index[0]
            nxt = c.index[c.index.get_loc(d) + 1] if c.index.get_loc(d) + 1 < len(c) else d
            P = k["open"].loc[nxt]
            av = (P - c.loc[tr]) / (c.loc[pk] - c.loc[tr])
            r[n] = f"{av:.0%}({(d - pk).days}일)"; avoid[n].append(max(av, 0.0))
        erows.append(r)
    print("[2] 큰 하락 구간별: 회피비율(고점 후 발동까지 달력일). '-'는 저점 전 발동 없음")
    print(pd.DataFrame(erows).to_string(index=False), "\n")
    summ = pd.DataFrame({n: {"평균 회피비율": np.mean(v), "회피 25%↑ 구간수": int(sum(x >= 0.25 for x in v)), "구간수": len(v)} for n, v in avoid.items()}).T
    print("[3] 요약 (회피비율 평균은 미발동=0%로 계산)"); print(pd.concat([summ, pd.DataFrame(rows).T[["유효율", "오경보"]]], axis=1).to_string(), "\n")

    # ---- 현재 상태
    print(f"[4] 현재({c.index[-1].date()}) 상태 — 고점 {c.max():.0f}({c.idxmax().date()}) 대비 {c.iloc[-1] / c.max() - 1:.1%}")
    cur_pk = c.idxmax()
    for n, s in rules.items():
        w = s.loc[cur_pk:]
        first = w[w & ~s.shift(1, fill_value=False).loc[cur_pk:]]
        txt = f"최초 발동 {first.index[0].date()} (고점 후 {(first.index[0] - cur_pk).days}일, 낙폭 {dd.loc[first.index[0]]:.0%})" if len(first) else "미발동"
        print(f"   {n}: {'발동 중' if s.iloc[-1] else '해제'} / {txt}")

    # ---- 차트
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EN = {"DD-10%": "DD-10%", "200일선 이탈(3일)": "Below MA200", "DD-10% & 200일선 아래": "DD-10% & below MA200"}
    show = ["DD-10%", "200일선 이탈(3일)", "DD-10% & 200일선 아래"]
    cols = ["#e03131", "#1971c2", "#2b8a3e"]
    fig, ax = plt.subplots(2, 1, figsize=(18, 11), gridspec_kw={"height_ratios": [3, 2]})
    cc = c.loc[start:]
    ax[0].semilogy(cc.index, cc, color="#1f3b73", lw=1)
    for _, e in base.iterrows():
        ax[0].axvspan(e["peak"], e["trough"], color="#e03131", alpha=0.10)
    for n, colr, mk in zip(show, cols, ["v", "s", "D"]):
        e = ev[n]
        good = e.fwd_min <= GOOD
        ax[0].scatter(e.date[good & e.complete], cc.loc[e.date[good & e.complete]], marker=mk, color=colr, s=70, label=f"{EN[n]} (valid)", zorder=5)
        bad = (~good) & e.complete
        ax[0].scatter(e.date[bad], cc.loc[e.date[bad]], marker=mk, facecolors="none", edgecolors=colr, s=70, label=f"{EN[n]} (false alarm)", zorder=5)
    ax[0].legend(loc="upper left", ncol=3, fontsize=8); ax[0].set_title("Trigger events (filled = further drop >=10% within 120d, hollow = no big drop after)")
    ax[0].grid(alpha=0.25)
    w = 0.25; x = np.arange(len(base))
    for j, (n, colr) in enumerate(zip(show, cols)):
        vals = [max(float(str(r[n]).split("%")[0]) / 100, 0) if r[n] != "-" else 0 for r in erows]
        ax[1].bar(x + (j - 1) * w, vals, w, color=colr, label=EN[n])
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{r['고점']}\n{r['낙폭']:.0%}" for r in erows], fontsize=8)
    ax[1].set_ylim(0, 1); ax[1].set_title("Share of peak-to-trough drop avoided"); ax[1].legend(); ax[1].grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "decline_response.png", dpi=110)
    print(f"\n차트: {OUT / 'decline_response.png'}")


if __name__ == "__main__":
    main()
