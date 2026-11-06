"""다종목 포트폴리오 엔진. 목표 비중(wide DataFrame)을 받아 다음날 시가 체결로 계산.

weights: index=신호일(종가 기준), columns=종목코드, 값=포트폴리오 대비 비중(롱 +, 숏 -).
"""
import pandas as pd

from engine import Costs


def run(open_: pd.DataFrame, close: pd.DataFrame, weights: pd.DataFrame,
        costs: Costs = Costs(), borrow_rate: float = 0.04) -> pd.DataFrame:
    w = weights.reindex(index=close.index, columns=close.columns).fillna(0)
    pos = w.shift(1).fillna(0)
    prev = pos.shift(1).fillna(0)
    overnight = (open_ / close.shift(1) - 1).fillna(0)
    intraday = (close / open_ - 1).fillna(0)
    gross = (prev * overnight + pos * intraday).sum(axis=1)

    trade = (pos - prev).abs()
    # 매도(롱 청산 + 신규 숏)에만 거래세
    sell = (prev.clip(lower=0) - pos.clip(lower=0)).clip(lower=0) + (prev.clip(upper=0) - pos.clip(upper=0)).clip(upper=0).abs()
    cost = (trade * (costs.fee + costs.slippage) + sell * costs.tax).sum(axis=1)
    borrow = pos.clip(upper=0).abs().sum(axis=1) * borrow_rate / 252
    ret = gross - cost - borrow
    out = pd.DataFrame({"ret": ret, "gross_exposure": pos.abs().sum(axis=1),
                        "net_exposure": pos.sum(axis=1)})
    out["equity"] = (1 + ret).cumprod()
    return out
