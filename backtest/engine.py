"""벡터화 백테스트 엔진 (롱온리, 다음날 시가 체결).

전략은 종가 기준으로 신호(1=보유, 0=현금)를 낸다 -> 다음 거래일 시가에 체결.
- 전일 종가->당일 시가 구간 수익은 '전일 포지션'이, 시가->종가 구간은 '당일 포지션'이 가져간다.
- 비용: 매수 = 수수료+슬리피지, 매도 = 수수료+거래세+슬리피지 (편도, 비율)
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Costs:
    fee: float = 0.00015   # 증권사 수수료(편도)
    tax: float = 0.0018    # 매도 거래세 (시기별로 다르니 확인해서 조정)
    slippage: float = 0.0005


def run(df: pd.DataFrame, signal: pd.Series, costs: Costs = Costs()) -> pd.DataFrame:
    sig = signal.reindex(df.index).fillna(0).clip(0, 1)
    pos = sig.shift(1).fillna(0)           # 오늘 시가에 체결되는 포지션(어제 신호)
    prev = pos.shift(1).fillna(0)          # 어제 보유 포지션
    overnight = df["open"] / df["close"].shift(1) - 1
    intraday = df["close"] / df["open"] - 1
    d = pos - prev
    cost = d.clip(lower=0) * (costs.fee + costs.slippage) + (-d).clip(lower=0) * (costs.fee + costs.tax + costs.slippage)
    ret = prev * overnight.fillna(0) + pos * intraday - cost
    out = pd.DataFrame({"pos": pos, "ret": ret})
    out["equity"] = (1 + ret).cumprod()
    out["bh_equity"] = (df["close"] / df["close"].iloc[0])
    return out
