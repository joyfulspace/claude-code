"""지표의 원천(raw) 시계열을 KOSPI 와 함께 그림(한글): python3 backtest/explain_raw_series.py"""
import sys
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import risk_exit as rx  # noqa: E402

OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, GREEN, BLUE, ORANGE, GRAY = "#e03131", "#2b8a3e", "#1971c2", "#f08c00", "#868e96"


def main():
    k = pd.read_csv(data.DATA_DIR / "KOSPI_ohlcv.csv", index_col=0, parse_dates=True); c = k["close"]
    f = pd.read_csv(data.DATA_DIR / "KOFIA_funds.csv", index_col=0, parse_dates=True)
    S = {n: data.load_series(n) for n in ("VIX", "SPX", "USDKRW", "TNX", "KOSDAQ")}
    ep = rx.drawdown_episodes(c, depth=-0.15)
    dep, mar = f["deposit"] / 1e6, f["margin"] / 1e6                      # 조원
    ratio = (f["margin"] / f["deposit"]) * 100
    forced = f["forced_ratio"]
    kq = (S["KOSDAQ"] / c.reindex(S["KOSDAQ"].index)).dropna()
    kq = kq / kq.iloc[0] * 100
    series = {
        "deposit": ("투자자예탁금 (조원) — 증시 대기 매수 여력", dep, GREEN, None),
        "margin": ("신용융자 잔고 (조원) — 빚내서 산 주식", mar, GREEN, None),
        "ratio": ("신용융자 ÷ 투자자예탁금 (%) — 매수 여력 대비 레버리지", ratio, GREEN, None),
        "forced": ("반대매매 비중 (%) — 연한 선=일별, 진한 선=20일 평균", forced, GREEN, 20),
        "fx": ("원/달러 환율 (원) — 위로 갈수록 원화 약세", S["USDKRW"], ORANGE, None),
        "spx": ("S&P500 (로그)", S["SPX"], BLUE, None),
        "tnx": ("미국 10년 국채금리 (%)", S["TNX"], BLUE, None),
        "vix": ("VIX (공포지수)", S["VIX"], ORANGE, None),
        "kq": ("코스닥 ÷ 코스피 (2000-10=100) — 내려가면 중소형 위험회피", kq, ORANGE, None),
    }

    def draw(keys, fname, title, t0=None, t1=None, linear=False, h=3.0):
        n = len(keys) + 1
        fig, axes = plt.subplots(n, 1, figsize=(19, 3.0 + h * len(keys)), sharex=True, gridspec_kw={"height_ratios": [2.4] + [1.4] * len(keys)})
        t0 = t0 or c.index[0]; t1 = t1 or c.index[-1]
        cc = c.loc[t0:t1]
        (axes[0].plot if linear else axes[0].semilogy)(cc.index, cc, color="#1f3b73", lw=1.4)
        axes[0].set_ylabel("KOSPI"); axes[0].set_title(title, fontsize=15)
        for pk, tr, depth in [(e.peak, e.trough, e.depth) for e in ep.itertuples()]:
            if tr >= t0 and pk <= t1:
                for a in axes: a.axvspan(max(pk, t0), min(tr, t1), color=RED, alpha=0.10)
                axes[0].annotate(f"{depth:.0%}", (max(pk, t0), c.loc[pk] if pk >= t0 else cc.iloc[0]), textcoords="offset points", xytext=(0, 8), color=RED, fontsize=10.5, fontweight="bold", ha="center")
        for a, key in zip(axes[1:], keys):
            lab, s, col, roll = series[key]; s = s.loc[t0:t1]
            if roll:
                a.plot(s.index, s, color=col, lw=0.6, alpha=0.35); a.plot(s.index, s.rolling(roll).mean(), color=col, lw=2.0)
            else:
                (a.semilogy if key == "spx" else a.plot)(s.index, s, color=col, lw=1.6)
            a.set_title(lab, loc="left"); a.grid(alpha=0.25)
            last = s.dropna()
            if len(last): a.annotate(f"{last.iloc[-1]:,.1f}", (last.index[-1], last.iloc[-1]), textcoords="offset points", xytext=(6, 0), fontsize=10.5, color=col, fontweight="bold")
        axes[0].grid(alpha=0.25)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2 if not linear else 1)); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if linear:
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3)); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=105); plt.close(fig)

    draw(["deposit", "margin", "ratio", "forced", "fx"], "raw_1_수급과환율.png", "KOSPI와 수급·환율 원본 시계열 (2000~, 빨간 음영 = 고점 대비 -15% 이상 하락 구간)")
    draw(["spx", "tnx", "vix", "kq"], "raw_2_미국과심리.png", "KOSPI와 미국시장·심리 원본 시계열 (2000~)")
    draw(["deposit", "margin", "ratio", "forced", "fx", "tnx", "vix", "kq"], "raw_3_최근2년.png", "최근 2년 확대 (2024-10~): 모든 원본 시계열", pd.Timestamp("2024-10-01"), None, True, 2.2)

    pk = c.loc["2026-01-01":].idxmax(); now = c.index[-1]
    print(f"KOSPI 고점 {pk.date()} {c.loc[pk]:,.0f} → 현재 {now.date()} {c.iloc[-1]:,.0f} ({c.iloc[-1] / c.loc[pk] - 1:.0%})")
    for key in ("deposit", "margin", "ratio", "fx", "tnx", "vix"):
        s = series[key][1].dropna(); a = s.loc[:pk].iloc[-1]; b = s.iloc[-1]
        print(f"{series[key][0].split(' — ')[0]:<26} 고점일 {a:>10,.2f} → 현재 {b:>10,.2f}  | 전체기간 최대 {s.max():,.2f} ({s.idxmax().date()})")
    fr = forced.loc["2026-01-01":]; print(f"반대매매 비중 2026년 평균 {fr.mean():.2f}% (전체 평균 {forced.mean():.2f}%), 2026 최대 {fr.max():.1f}% ({fr.idxmax().date()})")


if __name__ == "__main__":
    main()
