"""예시 규칙: 단기 이동평균이 장기 이동평균 위에 있으면 보유."""
import pandas as pd

PARAMS = {"fast": 20, "slow": 60}


def signal(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    return (df["close"].rolling(fast).mean() > df["close"].rolling(slow).mean()).astype(int)
