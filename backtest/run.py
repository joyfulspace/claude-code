"""사용: python3 backtest/run.py --strategy ma_cross --code 005930 --start 2020-01-01 --end 2025-12-31 [--param fast=10]

--code SYNTH 는 합성 데이터로 파이프라인만 점검(실제 성과 아님).
--split 2023-01-01 을 주면 그 날짜 전/후(in/out-of-sample)를 따로 보여준다.
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import engine  # noqa: E402
import metrics  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--param", action="append", default=[], help="key=value (숫자)")
    p.add_argument("--split")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()

    mod = importlib.import_module(f"strategies.{a.strategy}")
    params = {**getattr(mod, "PARAMS", {}), **{k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in a.param)}}
    df = data.load(a.code, a.start, a.end, a.refresh)
    res = engine.run(df, mod.signal(df, **params))

    import pandas as pd
    pd.options.display.float_format = "{:.3f}".format
    print(f"{a.strategy} {params} | {a.code} | {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}일)")
    if a.split:
        for name, part in (("IN-SAMPLE", res.loc[:a.split]), ("OUT-OF-SAMPLE", res.loc[a.split:])):
            part = part.copy()
            part["equity"] = (1 + part["ret"]).cumprod()
            part["bh_equity"] = df["close"].reindex(part.index) / df["close"].reindex(part.index).iloc[0]
            print(f"\n[{name}]\n{metrics.summarize(part)}")
    else:
        print(metrics.summarize(res))
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / f"{a.strategy}_{a.code}.csv")


if __name__ == "__main__":
    main()
