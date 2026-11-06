import numpy as np
import pandas as pd


def _stats(ret: pd.Series, equity: pd.Series) -> dict:
    years = max((ret.index[-1] - ret.index[0]).days / 365.25, 1e-9)
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    mdd = (equity / equity.cummax() - 1).min()
    return {"누적수익률": equity.iloc[-1] - 1, "CAGR": cagr, "변동성": vol,
            "샤프": (ret.mean() * 252) / vol if vol > 0 else np.nan, "MDD": mdd}


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    s = _stats(res["ret"], res["equity"])
    trades = int((res["pos"].diff().abs() > 0).sum())
    s["거래횟수(편도)"] = trades
    s["시장노출"] = res["pos"].mean()
    bh_ret = res["bh_equity"].pct_change().fillna(0)
    b = _stats(bh_ret, res["bh_equity"])
    return pd.DataFrame({"전략": pd.Series(s), "Buy&Hold": pd.Series(b)})
