"""사용: python3 backtest/run_portfolio.py --synth [--split 2018-01-01] [--param thresh=-0.05]
실데이터: data/ 에 KOSPI.csv, SPX.csv, valuation_pbr.csv, valuation_per.csv 와 --codes 의 종목 CSV 필요.
PBR / PER / 조합 세 가지를 기준(균등 보유)과 함께 한 표로 비교한다.
"""
import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import metrics  # noqa: E402
import portfolio  # noqa: E402

KINDS = ["pbr", "per", "combo"]


def conv(v):
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def table(results: dict, bench: pd.Series, sl) -> pd.DataFrame:
    cols = {}
    for name, res in results.items():
        r = res["ret"].loc[sl]
        cols[name] = metrics._stats(r, (1 + r).cumprod())
    b = bench.loc[sl].pct_change().fillna(0)
    cols["KOSPI B&H"] = metrics._stats(b, (1 + b).cumprod())
    return pd.DataFrame(cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="crash_value")
    p.add_argument("--synth", action="store_true")
    p.add_argument("--codes")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--split")
    p.add_argument("--param", action="append", default=[])
    a = p.parse_args()
    pd.options.display.float_format = "{:.3f}".format

    mod = importlib.import_module(f"strategies.{a.strategy}")
    params = {**mod.PARAMS, **{k: conv(v) for k, v in (x.split("=") for x in a.param)}}
    if a.synth:
        open_, close, kospi, spx, pbr, per = data.synthetic_panel()
    else:
        dfs = {c: data.load(c, a.start, a.end) for c in a.codes.split(",")}
        open_ = pd.DataFrame({c: d["open"] for c, d in dfs.items()})
        close = pd.DataFrame({c: d["close"] for c, d in dfs.items()})
        kospi, spx = data.load_series("KOSPI"), data.load_series("SPX")
        pbr, per = data.load_valuation("pbr"), data.load_valuation("per")

    crash = mod.regime(close.index, kospi, spx, params["n_months"], params["thresh"])
    ep = int((crash & ~crash.shift(1, fill_value=False)).sum())
    print(f"{a.strategy} {params}\n종목 {close.shape[1]}개, {close.index[0].date()}~{close.index[-1].date()}")
    print(f"동시 급락일 {int(crash.sum())}일 / 에피소드 {ep}회  <- 표본이 적으면 결과를 믿기 어려움")

    results = {"기준(균등보유)": portfolio.run(open_, close, mod.baseline(close))}
    for k in KINDS:
        val = mod.make_valuation(pbr, per, k)
        results[k.upper()] = portfolio.run(open_, close, mod.weights(close, kospi, spx, val, **params))
    bench = kospi.reindex(close.index).ffill()

    parts = [("전체", slice(None))] if not a.split else [("IN-SAMPLE", slice(None, a.split)), ("OUT-OF-SAMPLE", slice(a.split, None))]
    for label, sl in parts:
        t = table(results, bench, sl)
        t.loc["CAGR 초과(vs 기준)"] = [c - t.loc["CAGR", "기준(균등보유)"] for c in t.loc["CAGR"]]
        print(f"\n[{label}]\n{t}")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame({k: v["ret"] for k, v in results.items()}).to_csv(out / f"{a.strategy}_portfolio.csv")


if __name__ == "__main__":
    main()
