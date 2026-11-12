"""국내 주식 일봉 로더: CSV 캐시 -> (없으면) 공공데이터포털 API -> (테스트용) 합성 데이터."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
API_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
COLS = ["open", "high", "low", "close", "volume"]


def _fetch_api(code: str, start: str, end: str) -> pd.DataFrame:
    import requests

    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        raise RuntimeError("환경변수 DATA_GO_KR_KEY 가 없습니다.")
    rows, page = [], 1
    while True:
        r = requests.get(API_URL, params={
            "serviceKey": key, "resultType": "json", "numOfRows": 1000, "pageNo": page,
            "likeSrtnCd": code, "beginBasDt": start.replace("-", ""), "endBasDt": end.replace("-", ""),
        }, timeout=30)
        body = r.json()
        if "response" not in body:
            raise RuntimeError(f"API 오류: {body}")
        b = body["response"]["body"]
        items = b.get("items", {}).get("item", []) if b.get("totalCount") else []
        rows += items if isinstance(items, list) else [items]
        if page * 1000 >= int(b.get("totalCount", 0)):
            break
        page += 1
    df = pd.DataFrame(rows)
    df = df[df["srtnCd"] == code]  # likeSrtnCd 는 부분일치라 정확히 필터
    df.index = pd.to_datetime(df["basDt"])
    df = df.rename(columns={"mkp": "open", "hipr": "high", "lopr": "low", "clpr": "close", "trqu": "volume"})
    return df[COLS].astype(float).sort_index()


def synthetic(days: int = 1500, seed: int = 0) -> pd.DataFrame:
    """파이프라인 점검용 가짜 데이터(실제 성과 아님)."""
    rng = np.random.default_rng(seed)
    close = 50000 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, days)))
    open_ = close * (1 + rng.normal(0, 0.004, days))
    idx = pd.bdate_range("2019-01-01", periods=days)
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.005,
                         "low": np.minimum(open_, close) * 0.995, "close": close,
                         "volume": rng.integers(1e5, 1e6, days).astype(float)}, index=idx)


def load(code: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """code: 6자리 종목코드(예: 005930). data/{code}.csv 에 캐시."""
    if code == "SYNTH":
        return synthetic()
    path = DATA_DIR / f"{code}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = _fetch_api(code, start, end)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(path)
    return df.loc[start:end]


def load_series(name: str) -> pd.Series:
    """지수 종가: data/KOSPI.csv, data/SPX.csv (컬럼: date, close)."""
    return pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)["close"]


def load_valuation(kind: str) -> pd.DataFrame:
    """data/valuation_pbr.csv, valuation_per.csv: 행=날짜, 열=종목코드. 공시 시점 이후 값만 넣을 것."""
    return pd.read_csv(DATA_DIR / f"valuation_{kind}.csv", index_col=0, parse_dates=True)


def synthetic_panel(n_stocks: int = 40, days: int = 2500, seed: int = 1):
    """파이프라인 점검용: (open, close, kospi, spx, valuation). 실제 성과 아님."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-01", periods=days)
    g = rng.normal(0.0003, 0.008, days)           # 글로벌 공통 요인
    kr = g + rng.normal(0, 0.006, days)
    us = g + rng.normal(0, 0.006, days)
    for s in (600, 1300, 2000):                    # 동시 급락 구간 주입
        kr[s:s + 40] -= 0.004
        us[s:s + 40] -= 0.004
    kospi = pd.Series(2000 * np.exp(np.cumsum(kr)), idx)
    spx = pd.Series(2000 * np.exp(np.cumsum(us)), idx)
    codes = [f"{i:06d}" for i in range(n_stocks)]
    beta = rng.uniform(0.7, 1.4, n_stocks)
    r = kr[:, None] * beta + rng.normal(0, 0.012, (days, n_stocks))
    close = pd.DataFrame(10000 * np.exp(np.cumsum(r, axis=0)), idx, codes)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, (days, n_stocks)))
    def _val(sd):
        return pd.DataFrame(np.exp(rng.normal(0, 0.5, n_stocks))[None, :] * np.exp(rng.normal(0, 0.05, (days, n_stocks)).cumsum(0) * 0.05),
                            idx, codes)
    return open_, close, kospi, spx, _val(1), _val(2)


YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
INDEX_SYMBOLS = {"SPX": "^GSPC", "KOSPI": "^KS11", "VIX": "^VIX"}


def fetch_yahoo_index(name: str, start: str = "2000-01-01", end: str | None = None) -> pd.Series:
    """야후 파이낸스 차트 API로 지수 일봉 종가를 받아 data/{name}.csv 로 저장 (yfinance 불필요)."""
    import requests

    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end or pd.Timestamp.now().normalize() + pd.Timedelta(days=1), tz="UTC").timestamp())
    r = requests.get(YAHOO_URL.format(sym=INDEX_SYMBOLS[name]),
                     params={"period1": p1, "period2": p2, "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=idx, name="close").dropna()
    s = s[~s.index.duplicated(keep="last")]
    DATA_DIR.mkdir(exist_ok=True)
    s.rename_axis("date").to_csv(DATA_DIR / f"{name}.csv")
    return s


def fetch_yahoo_ohlcv(name: str = "KOSPI", start: str = "2000-01-01") -> pd.DataFrame:
    """지수 OHLCV 를 받아 data/{name}_ohlcv.csv 로 저장 (거래량 지표용)."""
    import requests

    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)).timestamp())
    r = requests.get(YAHOO_URL.format(sym=INDEX_SYMBOLS[name]),
                     params={"period1": p1, "period2": p2, "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    df = pd.DataFrame(res["indicators"]["quote"][0], index=idx)[COLS].dropna()
    df = df[~df.index.duplicated(keep="last")]
    DATA_DIR.mkdir(exist_ok=True)
    df.rename_axis("date").to_csv(DATA_DIR / f"{name}_ohlcv.csv")
    return df
