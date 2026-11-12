"""추세 이탈 × 수급 조합 비교: python3 backtest/run_trend_supply.py

규칙(결과를 보기 전에 고정, 모두 종가 기준 신호 → 다음날 시가 체결, 3일 연속 확인):
  T1  추세: 종가 < 200일선 이면 매도, > 200일선 이면 재진입
  T2  추세: 50일선 < 200일선(데드크로스) 이면 매도, 반대면 재진입
  S   수급: 수급 위험 백분위 >= 80 이면 매도, <= 50 이면 재진입
  C1  조합(AND): 종가<200일선 AND 수급>=70 이면 매도, 종가>200일선 이면 재진입
  C2  조합(동적): 수급>=70 이면 50일선, 아니면 200일선을 기준선으로 이탈/재진입
수급 위험 = 신용융자 증가율·신용융자/예탁금·예탁금 감소·반대매매 비중의 평균 → 과거 대비 백분위
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import risk_exit as rx  # noqa: E402
from run_risk_exit import backtest, stats  # noqa: E402

OUT = Path(__file__).parent / "results"
CONFIRM = 3


def hysteresis(exit_cond: pd.Series, enter_cond: pd.Series, days: int = CONFIRM) -> pd.Series:
    """exit_cond 가 days일 연속이면 0(현금), enter_cond 가 days일 연속이면 1(보유)."""
    ex = exit_cond.fillna(False).astype(int).rolling(days).sum() >= days
    en = enter_cond.fillna(False).astype(int).rolling(days).sum() >= days
    state, out = 1, []
    for x, e in zip(ex.values, en.values):
        if state == 1 and x:
            state = 0
        elif state == 0 and e:
            state = 1
        out.append(state)
    return pd.Series(out, index=exit_cond.index, dtype=float)


def build_signals(k, supply_pct, ma_long=200, ma_fast=50):
    c = k["close"]
    ma_l, ma_f = c.rolling(ma_long).mean(), c.rolling(ma_fast).mean()
    sp = supply_pct
    sig = {}
    sig["T1 200일선 이탈"] = hysteresis(c < ma_l, c > ma_l)
    sig["T2 데드크로스"] = hysteresis(ma_f < ma_l, ma_f > ma_l)
    sig["S 수급 단독"] = hysteresis(sp >= 80, sp <= 50)
    sig["C1 이탈 AND 수급"] = hysteresis((c < ma_l) & (sp >= 70), c > ma_l)
    ref = pd.Series(np.where(sp >= 70, ma_f, ma_l), index=c.index)
    sig["C2 수급따라 기준선 변경"] = hysteresis(c < ref, c > ref)
    return sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="2013-01-01")
    a = p.parse_args()
    pd.options.display.float_format = "{:.3f}".format
    OUT.mkdir(exist_ok=True)

    k = pd.read_csv(data.DATA_DIR / "KOSPI_ohlcv.csv", index_col=0, parse_dates=True)
    vix = data.load_series("VIX")
    funds = pd.read_csv(data.DATA_DIR / "KOFIA_funds.csv", index_col=0, parse_dates=True)
    comp = rx.components(k, vix, funds)
    sup_cols = [c for c in comp.columns if c.startswith("supply_")]
    supply_pct = rx.pct_rank(comp[sup_cols].mean(axis=1, skipna=False), 252) * 100
    start = supply_pct.first_valid_index()

    sigs = build_signals(k, supply_pct)
    rets = {"Buy&Hold": backtest(k, pd.Series(1.0, index=k.index))["ret"]}
    for n, s in sigs.items():
        s = s.copy()
        s[:start] = 1.0
        sigs[n] = s
        rets[n] = backtest(k, s)["ret"]
    R = pd.DataFrame(rets).loc[start:]
    print(f"비교 구간 {start.date()} ~ {k.index[-1].date()} | 신호 → 다음날 시가 체결, ETF 가정(거래세 0), 3일 연속 확인\n")

    def block(sl, label):
        cols = {}
        for n in R.columns:
            s = stats(R[n].loc[sl], sigs[n].reindex(R.loc[sl].index) if n in sigs else None)
            cols[n] = s
        print(f"[{label}]\n{pd.DataFrame(cols)}\n")
    block(slice(None), "전체")
    block(slice(None, a.split), f"IN-SAMPLE (~{a.split})")
    block(slice(a.split, None), f"OUT-OF-SAMPLE ({a.split}~)")

    # --- 큰 하락 구간에서 전략별 손실 (고점→저점)
    eq = (1 + R).cumprod()
    ep = rx.drawdown_episodes(k["close"].loc[start:])
    rows = []
    for _, e in ep.iterrows():
        row = {"고점": e["peak"].date(), "저점": e["trough"].date()}
        for n in eq.columns:
            row[n] = eq[n].loc[e["trough"]] / eq[n].loc[e["peak"]] - 1
        rows.append(row)
    print("[-15% 이상 하락 구간: 고점→저점 구간 손익]")
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    # --- 현재 상태
    print("[현재 신호]", {n: ("현금" if s.iloc[-1] == 0 else "보유") for n, s in sigs.items()}, f"| 수급 위험 백분위 {supply_pct.dropna().iloc[-1]:.0f}")
    for n, s in sigs.items():
        ch = s.index[s.diff().abs() == 1]
        print(f"  {n}: 최근 전환 {[f'{d.date()}({"매도" if s.loc[d]==0 else "재진입"})' for d in ch[-3:]]}")

    # --- 민감도 (T1/C1 의 이동평균 길이)
    print("\n[민감도: 이동평균 길이별 전체 CAGR / MDD]")
    sens = {}
    for L in (100, 150, 200, 250):
        s = build_signals(k, supply_pct, ma_long=L)
        for n in ("T1 200일선 이탈", "C1 이탈 AND 수급"):
            q = s[n].copy(); q[:start] = 1.0
            st = stats(backtest(k, q)["ret"].loc[start:])
            sens.setdefault(n, {})[f"MA{L}"] = f"{st['CAGR']:.3f} / {st['MDD']:.3f}"
    print(pd.DataFrame(sens).T.to_string())

    # --- 차트
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(sigs)
    fig, ax = plt.subplots(3, 1, figsize=(18, 13), sharex=True, gridspec_kw={"height_ratios": [3, 2.2, 2.4]})
    c = k["close"].loc[start:]
    ax[0].semilogy(c.index, c, color="#1f3b73", lw=1)
    ax[0].set_title("KOSPI (log)")
    cols = ["#e03131", "#f08c00", "#2b8a3e", "#9c36b5", "#1971c2"]
    for i, n in enumerate(names):
        s = sigs[n].loc[start:]
        ax[1].fill_between(s.index, i, i + 0.8, where=(s == 0).values, step="post", color=cols[i], alpha=0.8)
    ax[1].set_yticks([i + 0.4 for i in range(len(names))]); ax[1].set_yticklabels([n.split(" ")[0] for n in names])
    ax[1].set_title("Out-of-market periods (T1 close<MA200 / T2 death cross / S supply-only / C1 trend AND supply / C2 dynamic MA)")
    for n, colr in zip(eq.columns, ["gray"] + cols):
        ax[2].semilogy(eq.index, eq[n], label=n.split(" ")[0] if n != "Buy&Hold" else n, color=colr, lw=1)
    ax[2].legend(loc="upper left", ncol=3); ax[2].set_title("Equity (log, after costs)")
    for x in ax: x.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "trend_supply.png", dpi=110)
    pd.concat({n: s for n, s in sigs.items()}, axis=1).assign(close=k["close"]).to_csv(OUT / "trend_supply_signals.csv")
    print(f"\n차트: {OUT / 'trend_supply.png'}")


if __name__ == "__main__":
    main()
