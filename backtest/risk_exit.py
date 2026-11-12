"""KOSPI 위험 점수(0~100)와 전액 매도 알림.

범주(각 0~100, 동일 가중 평균): 시장 국면 / 심리 / 거래량 / (수급: 데이터 확보 후 추가 예정)
- 모든 지표는 '오늘까지의 과거 분포' 대비 백분위(확장 윈도우)라 미래 정보를 쓰지 않는다.
- VIX 는 미국 종가가 한국 마감 뒤에 확정되므로 1거래일 지연해서 사용.
"""
import numpy as np
import pandas as pd

WARMUP = 504  # 백분위 계산에 필요한 최소 관측치(약 2년)


def pct_rank(s: pd.Series, min_periods: int = WARMUP) -> pd.Series:
    def f(x):
        return (x[:-1] < x[-1]).mean() if np.isfinite(x[-1]) else np.nan
    return s.expanding(min_periods=min_periods).apply(f, raw=True)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def components(k: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """k: KOSPI OHLCV. 반환: 지표별 0~1 (클수록 위험)."""
    c, v = k["close"], k["volume"]
    vix = vix.reindex(k.index).ffill().shift(1)
    ma200, ma120 = c.rolling(200).mean(), c.rolling(120).mean()
    out = pd.DataFrame(index=k.index)
    # 시장 국면
    out["regime_gap200"] = pct_rank(c / ma200 - 1)
    out["regime_ret63"] = pct_rank(c.pct_change(63))
    out["regime_trend_break"] = ((c < ma120) & (ma120 < ma120.shift(20))).astype(float).where(ma120.notna())
    # 심리
    out["sent_vix_complacency"] = 1 - pct_rank(vix)
    out["sent_vix_jump"] = pct_rank(vix.pct_change(5))
    out["sent_rsi"] = pct_rank(rsi(c))
    # 거래량
    out["vol_surge"] = pct_rank(v.rolling(20).mean() / v.rolling(120).mean())
    near_high = (c / c.rolling(252).max() >= 0.95)
    out["vol_divergence"] = ((1 - pct_rank(v.rolling(20).mean() / v.rolling(60).mean())) * near_high).where(near_high.notna())
    return out


def score(comp: pd.DataFrame) -> pd.DataFrame:
    cats = {}
    for cat in ("regime", "sent", "vol"):
        cols = [c for c in comp.columns if c.startswith(cat + "_")]
        cats[cat] = comp[cols].mean(axis=1, skipna=False) * 100
    df = pd.DataFrame(cats)
    df["raw"] = df[["regime", "sent", "vol"]].mean(axis=1, skipna=False)
    # 평균을 내면 점수가 중간으로 몰리므로, 다시 '과거 대비 상위 몇 %' 로 환산(0~100)
    df["score"] = pct_rank(df["raw"], min_periods=252) * 100
    return df


def binary_signal(score_: pd.Series, hi: float = 75, lo: float = 50, days: int = 3) -> pd.Series:
    """1=보유, 0=전액 현금. score>=hi 가 days일 연속이면 매도, score<=lo 가 되면 재진입(히스테리시스)."""
    over = (score_ >= hi).astype(int).rolling(days).sum() >= days
    state, out = 1, []
    for o, s in zip(over.values, score_.values):
        if state == 1 and o:
            state = 0
        elif state == 0 and s <= lo:
            state = 1
        out.append(state)
    return pd.Series(out, index=score_.index, dtype=float)


def graded_exposure(score_: pd.Series, start: float = 50, end: float = 80) -> pd.Series:
    """점수 5일 평균이 start→end 로 오르면 비중 100%→0% (0.25 단위)."""
    s5 = score_.rolling(5).mean()
    e = ((end - s5) / (end - start)).clip(0, 1)
    return ((e * 4).round() / 4).fillna(1.0)


def drawdown_episodes(close: pd.Series, depth: float = -0.15) -> pd.DataFrame:
    peak = close.cummax()
    dd = close / peak - 1
    grp = (dd == 0).cumsum()
    rows = []
    for _, s in dd.groupby(grp):
        if s.min() <= depth:
            hit10 = s[s <= -0.10].index[0]
            rows.append({"peak": s.index[0], "trough": s.idxmin(), "depth": s.min(), "dd10_date": hit10})
    return pd.DataFrame(rows)
