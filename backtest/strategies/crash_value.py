"""한국·미국 지수가 동시에 급락하면 저평가 종목 매수 / 고평가 종목 매도.

- 급락: 최근 n개월 수익률이 한국(KOSPI)과 미국(S&P500) 모두 thresh 이하
- 미국 지수는 1일 지연(한국 장 마감 시점에 확정된 미국 종가만 사용)
- 진입: 급락 상태이고 현재 포지션이 없을 때, 밸류에이션 하위 quantile 매수 / 상위 quantile 매도
- 청산: hold_days 거래일 후 전량 청산(그때도 급락이면 재진입)
- mode: long_short(롱 0.5 / 숏 -0.5) 또는 long_only(롱 1.0, 고평가는 아예 사지 않음)
- valuation: 낮을수록 저평가 (PBR/PER 등). 0 이하 값은 제외.
"""
import pandas as pd

PARAMS = {"n_months": 3, "thresh": -0.10, "hold_days": 60, "quantile": 0.2, "mode": "long_short"}


def regime(close_idx: pd.Index, kospi: pd.Series, spx: pd.Series, n_months: int, thresh: float) -> pd.Series:
    n = int(n_months) * 21
    kr = kospi.reindex(close_idx).ffill()
    us = spx.reindex(close_idx).ffill().shift(1)
    return (kr.pct_change(n) <= thresh) & (us.pct_change(n) <= thresh)


def weights(close: pd.DataFrame, kospi: pd.Series, spx: pd.Series, valuation: pd.DataFrame,
            n_months=3, thresh=-0.10, hold_days=60, quantile=0.2, mode="long_short") -> pd.DataFrame:
    crash = regime(close.index, kospi, spx, n_months, thresh)
    val = valuation.reindex(index=close.index, columns=close.columns).ffill().where(lambda v: v > 0)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cur, left = None, 0
    for i, d in enumerate(close.index):
        if cur is not None:
            left -= 1
            if left <= 0:
                cur = None
        if cur is None and crash.iloc[i]:
            v = val.loc[d].dropna()
            k = int(len(v) * quantile)
            if k >= 1:
                r = v.sort_values()
                cheap, rich = r.index[:k], r.index[-k:]
                cur = pd.Series(0.0, index=close.columns)
                if mode == "long_short":
                    cur[cheap], cur[rich] = 0.5 / k, -0.5 / k
                else:
                    cur[cheap] = 1.0 / k
                left = int(hold_days)
        if cur is not None:
            w.iloc[i] = cur.values
    return w
