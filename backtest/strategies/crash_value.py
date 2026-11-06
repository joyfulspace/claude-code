"""한국·미국 지수가 동시에 급락하면, 보유 종목 중 고평가는 팔고 저평가는 더 산다 (숏 없음).

- 평소: 유니버스 균등 보유(기준 포트폴리오)
- 급락: 최근 n개월 수익률이 KOSPI, S&P500 모두 thresh 이하 (미국은 1일 지연해 사용)
- 급락 진입 시(rotate): 밸류에이션 상위 quantile 종목은 전량 매도, 그 대금을 하위 quantile 종목에 추가 매수
  (mode=cash_buy 는 평소 현금 보유 → 급락 때만 저평가 매수)
- hold_days 거래일 후 기준 포트폴리오로 복귀 (그때도 급락이면 재진입)
- 밸류에이션: pbr / per / combo(PBR·PER 순위 평균). 낮을수록 저평가, 0 이하 값은 제외
"""
import pandas as pd

PARAMS = {"n_months": 1, "thresh": -0.05, "hold_days": 60, "quantile": 0.2, "mode": "rotate"}


def make_valuation(pbr: pd.DataFrame, per: pd.DataFrame, kind: str) -> pd.DataFrame:
    pbr, per = pbr.where(pbr > 0), per.where(per > 0)
    if kind == "pbr":
        return pbr
    if kind == "per":
        return per
    if kind == "combo":
        per = per.reindex_like(pbr)
        return pd.concat([pbr.rank(axis=1, pct=True), per.rank(axis=1, pct=True)]).groupby(level=0).apply(
            lambda g: g.mean(skipna=False))
    raise ValueError(kind)


def regime(close_idx: pd.Index, kospi: pd.Series, spx: pd.Series, n_months: int, thresh: float) -> pd.Series:
    n = int(n_months) * 21
    kr = kospi.reindex(close_idx).ffill()
    us = spx.reindex(close_idx).ffill().shift(1)
    return (kr.pct_change(n) <= thresh) & (us.pct_change(n) <= thresh)


def baseline(close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0 / close.shape[1], index=close.index, columns=close.columns)


def weights(close: pd.DataFrame, kospi: pd.Series, spx: pd.Series, valuation: pd.DataFrame,
            n_months=1, thresh=-0.05, hold_days=60, quantile=0.2, mode="rotate") -> pd.DataFrame:
    crash = regime(close.index, kospi, spx, n_months, thresh)
    val = valuation.reindex(index=close.index, columns=close.columns).ffill()
    base = 1.0 / close.shape[1]
    idle = base if mode == "rotate" else 0.0
    w = pd.DataFrame(idle, index=close.index, columns=close.columns)
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
                if mode == "rotate":
                    cur = pd.Series(base, index=close.columns)
                    cur[rich] = 0.0
                    cur[cheap] += base * k / k      # 팔린 비중 합 = 사는 비중 합
                else:
                    cur = pd.Series(0.0, index=close.columns)
                    cur[cheap] = 1.0 / k
                left = int(hold_days)
        if cur is not None:
            w.iloc[i] = cur.values
    return w
