"""사용: python3 backtest/run_portfolio.py --synth [--param thresh=-0.08 --param mode=long_only]
실데이터: data/ 에 KOSPI.csv, SPX.csv, valuation.csv 와 --codes 의 종목 CSV 필요.
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


def conv(v):
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def report(res, bench, label):
    ret = res["ret"]
    eq = (1 + ret).cumprod()
    b = bench.pct_change().fillna(0)
    t = pd.DataFrame({"전략": metrics._stats(ret, eq), "KOSPI B&H": metrics._stats(b, (1 + b).cumprod())})
    t.loc["평균 노출(gross)"] = [res["gross_exposure"].mean(), float("nan")]
    print(f"\n[{label}]\n{t}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="crash_value")
    p.add_argument("--synth", action="store_true")
    p.add_argument("--codes", help="콤마 구분 종목코드")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--split")
    p.add_argument("--param", action="append", default=[])
    a = p.parse_args()
    pd.options.display.float_format = "{:.3f}".format

    mod = importlib.import_module(f"strategies.{a.strategy}")
    params = {**mod.PARAMS, **{k: conv(v) for k, v in (x.split("=") for x in a.param)}}
    if a.synth:
        open_, close, kospi, spx, val = data.synthetic_panel()
    else:
        dfs = {c: data.load(c, a.start, a.end) for c in a.codes.split(",")}
        open_ = pd.DataFrame({c: d["open"] for c, d in dfs.items()})
        close = pd.DataFrame({c: d["close"] for c, d in dfs.items()})
        kospi, spx, val = data.load_series("KOSPI"), data.load_series("SPX"), data.load_valuation()

    w = mod.weights(close, kospi, spx, val, **params)
    crash = mod.regime(close.index, kospi, spx, params["n_months"], params["thresh"])
    ep = int((crash & ~crash.shift(1, fill_value=False)).sum())
    print(f"{a.strategy} {params}\n종목 {close.shape[1]}개, {close.index[0].date()}~{close.index[-1].date()}")
    print(f"동시 급락일 {int(crash.sum())}일 / 에피소드 {ep}회  <- 표본이 적으면 결과를 믿기 어려움")
    res = portfolio.run(open_, close, w)
    bench = kospi.reindex(close.index).ffill()
    if a.split:
        report(res.loc[:a.split], bench.loc[:a.split], "IN-SAMPLE")
        report(res.loc[a.split:], bench.loc[a.split:], "OUT-OF-SAMPLE")
    else:
        report(res, bench, "전체")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / f"{a.strategy}_portfolio.csv")


if __name__ == "__main__":
    main()
