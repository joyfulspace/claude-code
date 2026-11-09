"""KOSPI, S&P500 일봉을 야후 파이낸스에서 받아 data/ 에 저장: python3 backtest/fetch_indices.py [시작일]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402

start = sys.argv[1] if len(sys.argv) > 1 else "2000-01-01"
for name in data.INDEX_SYMBOLS:
    s = data.fetch_yahoo_index(name, start)
    print(f"{name}: {len(s)}행 {s.index[0].date()} ~ {s.index[-1].date()}")
