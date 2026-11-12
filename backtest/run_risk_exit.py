"""KOSPI 전액 매도 알림 백테스트 + 차트: python3 backtest/run_risk_exit.py [--hi 75 --lo 50 --days 3]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import engine  # noqa: E402
import metrics  # noqa: E402
import portfolio  # noqa: E402
import risk_exit as rx  # noqa: E402

OUT = Path(__file__).parent / "results"


def backtest(k, expo):
    w = pd.DataFrame({"KOSPI": expo})
    res = portfolio.run(k[["open"]].rename(columns={"open": "KOSPI"}), k[["close"]].rename(columns={"close": "KOSPI"}), w,
                        costs=engine.Costs(fee=0.00015, tax=0.0, slippage=0.0005))  # ETF 가정: 거래세 면제
    return res


def stats(ret, expo_sig=None):
    eq = (1 + ret).cumprod()
    s = metrics._stats(ret, eq)
    if expo_sig is not None:
        s["평균노출"] = expo_sig.mean()
        s["비중변경횟수"] = float((expo_sig.diff().abs() > 0).sum())
    return s


def chart(k, sc, sig, res, alerts, reentries, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(18, 13), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.6]})
    c = k["close"]
    ax[0].semilogy(c.index, c, color="#1f3b73", lw=1)
    out = (sig == 0)
    ax[0].fill_between(c.index, c.min(), c.max(), where=out.values, color="#e03131", alpha=0.13, step="post", label="Out of market (binary)")
    ax[0].scatter(alerts, c.loc[alerts], marker="v", color="#e03131", s=90, zorder=5, label=f"SELL-ALL alert ({len(alerts)})")
    ax[0].scatter(reentries, c.loc[reentries], marker="^", color="#2b8a3e", s=60, zorder=5, label="Re-entry")
    ax[0].set_title("KOSPI with full-exit alerts (red = alert day, shaded = out of market)")
    ax[0].legend(loc="upper left")
    ax[1].plot(sc.index, sc["score"], color="black", lw=1, label="Risk score (percentile vs own history)")
    ax[1].axhline(args.hi, color="#e03131", ls="--", lw=0.8)
    ax[1].axhline(args.lo, color="#2b8a3e", ls="--", lw=0.8)
    ax[1].set_ylim(0, 100)
    ax[1].legend(loc="upper left")
    ax[1].set_title(f"Risk score (sell if >= {args.hi} for {args.days} days; re-enter at <= {args.lo})")
    for name, colr in (("Buy&Hold", "gray"), ("Binary exit", "#e03131"), ("Graded", "#1971c2")):
        ax[2].semilogy(res[name].index, res[name], label=name, color=colr, lw=1)
    ax[2].legend(loc="upper left")
    ax[2].set_title("Equity (start = 1, after costs)")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hi", type=float, default=90)
    p.add_argument("--lo", type=float, default=60)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--split", default="2013-01-01")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-supply", action="store_true", help="수급 지표 제외(비교용)")
    a = p.parse_args()
    pd.options.display.float_format = "{:.3f}".format
    OUT.mkdir(exist_ok=True)

    if a.refresh or not (data.DATA_DIR / "KOSPI_ohlcv.csv").exists():
        data.fetch_yahoo_ohlcv("KOSPI")
    if a.refresh or not (data.DATA_DIR / "VIX.csv").exists():
        data.fetch_yahoo_index("VIX")
    k = pd.read_csv(data.DATA_DIR / "KOSPI_ohlcv.csv", index_col=0, parse_dates=True)
    vix = data.load_series("VIX")
    funds = None
    if not a.no_supply:
        if a.refresh or not (data.DATA_DIR / "KOFIA_funds.csv").exists():
            data.fetch_kofia_funds()
        funds = pd.read_csv(data.DATA_DIR / "KOFIA_funds.csv", index_col=0, parse_dates=True)

    sc = rx.score(rx.components(k, vix, funds))
    sig = rx.binary_signal(sc["score"], a.hi, a.lo, a.days)
    sig[sc["score"].isna()] = 1.0
    gexp = rx.graded_exposure(sc["score"])
    gexp[sc["score"].isna()] = 1.0
    start = sc["score"].first_valid_index()

    r_bh = backtest(k, pd.Series(1.0, index=k.index))
    r_bin, r_gr = backtest(k, sig), backtest(k, gexp)
    res = pd.DataFrame({"Buy&Hold": r_bh["ret"], "Binary exit": r_bin["ret"], "Graded": r_gr["ret"]}).loc[start:]
    eq = (1 + res).cumprod()

    # --- 알림 시점
    alerts = sig.index[(sig.diff() == -1)]
    reent = sig.index[(sig.diff() == 1)]
    fwd_min = pd.Series([k["close"].iloc[i + 1:i + 61].min() / k["close"].iloc[i] - 1 if i + 1 < len(k) else np.nan
                         for i in range(len(k))], index=k.index)
    al = pd.DataFrame({"alert_date": alerts, "kospi": k["close"].loc[alerts].values, "score": sc["score"].loc[alerts].values,
                       "fwd60_min_return": fwd_min.loc[alerts].values})
    al["verdict"] = np.where(al["fwd60_min_return"] <= -0.10, "drop>=10%", np.where(al["fwd60_min_return"].isna(), "n/a", "no big drop"))
    al.to_csv(OUT / "risk_exit_alerts.csv", index=False)
    pd.concat([k["close"], sc, sig.rename("binary_hold"), gexp.rename("graded_exposure")], axis=1).to_csv(OUT / "risk_exit_signals.csv")

    print(f"유효 점수 구간: {start.date()} ~ {k.index[-1].date()}  (앞 {rx.WARMUP}일은 백분위 워밍업)")
    print(f"규칙: 위험점수 >= {a.hi} 가 {a.days}일 연속이면 전액 매도, <= {a.lo} 이면 재진입 / 임계값은 사전 설정(최적화 안 함)\n")

    # --- 성과 비교
    def block(sl, label):
        cols = {"Buy&Hold": stats(res["Buy&Hold"].loc[sl]),
                "Binary exit": stats(res["Binary exit"].loc[sl], sig.loc[res.loc[sl].index]),
                "Graded": stats(res["Graded"].loc[sl], gexp.loc[res.loc[sl].index])}
        print(f"[{label}]\n{pd.DataFrame(cols)}\n")
    block(slice(None), "전체"), block(slice(None, a.split), f"IN-SAMPLE (~{a.split})"), block(slice(a.split, None), f"OUT-OF-SAMPLE ({a.split}~)")

    # --- 폭락 포착
    ep = rx.drawdown_episodes(k["close"].loc[start:])
    rows = []
    for _, e in ep.iterrows():
        pre = alerts[(alerts >= e["peak"] - pd.Timedelta(days=90)) & (alerts <= e["dd10_date"])]
        first = pre[0] if len(pre) else pd.NaT
        rows.append({"고점": e["peak"].date(), "저점": e["trough"].date(), "낙폭": e["depth"],
                     "-10% 도달": e["dd10_date"].date(), "-10% 시점 이미 매도?": bool(sig.loc[e["dd10_date"]] == 0),
                     "첫 알림": first.date() if pd.notna(first) else "-",
                     "알림 시 고점대비": (k["close"].loc[first] / k["close"].loc[:first].max() - 1) if pd.notna(first) else np.nan})
    print("[-15% 이상 하락 구간별 포착 여부]")
    print(pd.DataFrame(rows).to_string(index=False))
    n_ok = sum(r["-10% 시점 이미 매도?"] for r in rows)
    ar = al[al["verdict"] != "n/a"]
    print(f"\n포착: {n_ok}/{len(rows)} 구간 | 알림 {len(al)}회 중 이후 60일 내 -10% 이상 하락 {int((ar['verdict'] == 'drop>=10%').sum())}회, "
          f"큰 하락 없음 {int((ar['verdict'] == 'no big drop').sum())}회 (오경보)")

    # --- 민감도
    print("\n[임계값 민감도: Binary exit, 전체 구간]")
    sens = {}
    for hi in (80, 85, 90, 95, 97):
        sg = rx.binary_signal(sc["score"], hi, a.lo, a.days)
        sg[sc["score"].isna()] = 1.0
        r = backtest(k, sg)["ret"].loc[start:]
        s = stats(r)
        sens[f"hi={hi}"] = {"CAGR": s["CAGR"], "MDD": s["MDD"], "샤프": s["샤프"], "알림횟수": int((sg.diff() == -1).sum())}
    print(pd.DataFrame(sens))

    # --- 현재 상태
    last = sc.dropna().iloc[-1]
    print(f"\n[현재 {sc.dropna().index[-1].date()}] 위험점수 {last['score']:.0f} (국면 {last['regime']:.0f} / 심리 {last['sent']:.0f} / 거래량 {last['vol']:.0f}"
          f"{'/ 수급 %.0f' % last['supply'] if 'supply' in last else ''}) → {'매도 구간(현금)' if sig.iloc[-1] == 0 else '보유 구간'}, 점진 비중 {gexp.iloc[-1]:.0%}")
    print("최근 알림:", [str(d.date()) for d in alerts[-5:]])

    chart(k.loc[start:], sc.loc[start:], sig.loc[start:], eq, alerts[alerts >= start], reent[reent >= start], a, OUT / ("risk_exit_nosupply.png" if a.no_supply else "risk_exit.png"))
    print(f"\n차트: {OUT / ("risk_exit_nosupply.png" if a.no_supply else "risk_exit.png")}\n알림 목록: {OUT / 'risk_exit_alerts.csv'}")


if __name__ == "__main__":
    main()
