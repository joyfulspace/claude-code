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
