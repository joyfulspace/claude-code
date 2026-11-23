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
INDEX_SYMBOLS = {"SPX": "^GSPC", "KOSPI": "^KS11", "VIX": "^VIX", "USDKRW": "KRW=X", "TNX": "^TNX", "KOSDAQ": "^KQ11"}


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


KOFIA_URL = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"


def _kofia(obj: str, start: str, end: str) -> pd.DataFrame:
    import requests

    r = requests.post(KOFIA_URL, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://freesis.kofia.or.kr/",
                                           "Content-Type": "application/json;charset=UTF-8"},
                      json={"dmSearch": {"tmpV40": "1000000", "tmpV41": "1", "tmpV1": "D", "tmpV45": start.replace("-", ""),
                                         "tmpV46": end.replace("-", ""), "OBJ_NM": obj}}, timeout=90)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["ds1"])
    df.index = pd.to_datetime(df["TMPV1"])
    return df.drop(columns="TMPV1").sort_index()


def fetch_kofia_funds(start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """금융투자협회 FreeSIS 증시자금동향 → data/KOFIA_funds.csv (단위: 백만원, forced_ratio 는 %).
    deposit=투자자예탁금, margin=신용융자 잔고(코스피+코스닥), unpaid=위탁매매 미수금, forced_ratio=미수금 대비 반대매매 비중.
    (컬럼 의미는 2020-03 예탁금 38조, 2021-09 신용융자 25.5조 등 알려진 값과 대조해 확인)"""
    end = end or pd.Timestamp.now().strftime("%Y%m%d")
    a = _kofia("STATSCU0100000060BO", start, end)
    b = _kofia("STATSCU0100000070BO", start, end)
    df = pd.DataFrame({"deposit": a["TMPV2"], "margin": b["TMPV2"], "unpaid": a["TMPV5"],
                       "forced_amount": a["TMPV6"], "forced_ratio": a["TMPV7"]}).astype(float)
    DATA_DIR.mkdir(exist_ok=True)
    df.rename_axis("date").to_csv(DATA_DIR / "KOFIA_funds.csv")
    return df


FRED_IDS = {"KR10Y": "IRLTLT01KRM156N", "KR3M": "IR3TIB01KRM156N", "KRBASE": "INTDSRKRM193N"}


def fetch_fred(name: str) -> pd.Series:
    """FRED 공개 CSV(월별) -> data/{name}.csv (date, close). 한국 10년 국채금리/3개월 은행간금리/재할인율."""
    import io

    import requests

    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": FRED_IDS[name]}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]), name="close").dropna()
    DATA_DIR.mkdir(exist_ok=True)
    s.rename_axis("date").to_csv(DATA_DIR / f"{name}.csv")
    return s


def monthly_to_daily(monthly: pd.Series, daily_index: pd.DatetimeIndex, lag_months: int = 2) -> pd.Series:
    """월별 값(월초 날짜)을 '발표 지연'을 반영해 일별로 확장: M월 값은 M+lag_months 월 1일부터 사용(미래 정보 방지)."""
    s = monthly.copy(); s.index = s.index + pd.DateOffset(months=lag_months)
    return s.reindex(s.index.union(daily_index)).ffill().reindex(daily_index)


# ---- 반대매매 비중 보정 (금융투자협회 통계의 집계 기준 변경: 대상금액 -> 체결금액, 2023-10-25 추정) ----
STUCK_EOK_0 = 4943.0      # 키움증권 영풍제지 미수금(공시, 2023-10-20): 반대매매 대상으로 남아 매일 중복 집계됨
STUCK_EOK_1 = 4333.0      # 거래재개 후 반대매매·회수 뒤 남은 미수채권(공시, 2023-11-06)
SPIKE = (pd.Timestamp("2023-10-18"), pd.Timestamp("2023-10-24"))
BREAK = pd.Timestamp("2023-10-25")
START_OK = pd.Timestamp("2006-04-14")


def forced_ratio_adjusted(funds: pd.DataFrame, cap_abs: float = 20.0, verbose: bool = False) -> pd.DataFrame:
    """반대매매 비중을 '현재(체결금액 기준) 수준'으로 보정한 시계열.
    1) 2006-04-14 이전: 미집계 -> NaN
    2) 2023-10-19~24: (반대매매 금액 - 4,943억)/(미수금 - 4,943억)  (거래정지로 체결 못 된 키움 대상금액 제거), 10/18 은 앞뒤 값으로 보간
    3) 2023-10-25~: 분모(미수금)에서 영풍제지 잔여 미수채권(4,943억→11/6 4,333억, 이후 유지)을 제거, 11/2~3 영풍제지 반대매매 체결분 급증은 정상 수준으로
    4) 과거(대상금액 기준) 값에 k 를 곱해 체결금액 기준 수준으로 환산: k = 사건 후 6개월(11/7~) 중앙값 / 사건 전 6개월 중앙값
    5) 상한 cap_abs(%): 복원 후에도 남을 수 있는 이상치용 안전장치. (기계적 분위 상한은 2026년 하락기의 실제 고점까지 깎아서 쓰지 않음)
    반환 컬럼: raw(원자료 %), adj(보정 %), basis('target'|'filled'|'spike'), unpaid_adj(억원)"""
    amt = funds["forced_amount"] / 100.0           # 억원/일
    unp = funds["unpaid"] / 100.0                   # 억원
    raw = funds["forced_ratio"].astype(float)
    idx = funds.index
    stuck = pd.Series(0.0, index=idx)
    after = idx >= BREAK
    days = (idx - BREAK).days.values
    stuck[after] = np.where(idx[after] <= pd.Timestamp("2023-11-06"),
                            STUCK_EOK_0 + (STUCK_EOK_1 - STUCK_EOK_0) * np.clip(days[after] / 12.0, 0, 1), STUCK_EOK_1)
    unp_adj = unp.copy(); unp_adj[after] = (unp - stuck)[after]
    ratio = raw.copy()
    ratio[idx < START_OK] = np.nan
    sp = (idx >= SPIKE[0]) & (idx <= SPIKE[1])
    fix = (idx >= pd.Timestamp("2023-10-19")) & (idx <= SPIKE[1])
    ratio[fix] = ((amt - STUCK_EOK_0) / (unp - STUCK_EOK_0) * 100)[fix]
    ratio[idx == pd.Timestamp("2023-10-18")] = np.nan
    ratio = ratio.interpolate(limit_area="inside")  # 10/18
    amt_adj = amt.copy()   # 영풍제지 반대매매 체결분(키움 11/6 공시: 약 610억 회수) = 11/2~3 의 일시적 급증 -> 직전 정상 수준으로
    base_amt = amt.loc["2023-10-25":"2023-11-01"].median()
    for d_ in ("2023-11-02", "2023-11-03"):
        if pd.Timestamp(d_) in amt_adj.index:
            amt_adj.loc[d_] = base_amt
    post = (amt_adj / unp_adj * 100).where(after)
    pre_med = ratio.loc["2023-04-18":"2023-10-17"].median()
    post_med = post.loc["2023-11-07":"2024-05-06"].median()
    k = post_med / pre_med
    adj = ratio.copy()
    before = idx < BREAK
    adj[before] = ratio[before] * k
    adj[after] = post[after]
    cap = float(cap_abs); adj = adj.clip(upper=cap)
    basis = pd.Series(np.where(sp, "spike", np.where(after, "filled", "target")), index=idx)
    out = pd.DataFrame({"raw": raw, "adj": adj, "basis": basis, "unpaid_adj": unp_adj})
    out.attrs.update({"k": float(k), "pre_med": float(pre_med), "post_med": float(post_med), "cap": float(cap)})
    if verbose:
        print(f"환산계수 k = {k:.3f} (사건 전 6개월 중앙값 {pre_med:.2f}% → 사건 후 6개월 {post_med:.2f}%), 상한 {cap:.1f}%")
    return out
