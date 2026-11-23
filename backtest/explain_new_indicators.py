"""새로 추가한 지표(신용융자 과열, 한국 금리, 한미 금리차, 환율·금리 결합) 원본·임계선·신호 그림(한글)"""
import sys
import warnings
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
from avoid_score import Scorer  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREEN, ORANGE, GRAY, PURPLE = "#e03131", "#1971c2", "#2b8a3e", "#f08c00", "#868e96", "#9c36b5"


def main():
    k, ser, funds = load_all(); S = Scorer(k); c = k["close"]; idx = k.index
    pct = pd.read_pickle(OUT / "param_opt_pct.pkl")
    f = funds.reindex(idx).ffill().shift(1)
    lag = lambda s: s.reindex(idx).ffill().shift(1)
    tnx, fx = lag(ser["TNX"]), lag(ser["USDKRW"])
    kr10 = data.monthly_to_daily(ser["KR10Y"], idx); kr3 = data.monthly_to_daily(ser["KR3M"], idx)
    ratio = (f["margin"] / f["deposit"]) * 100
    spread = kr10 - tnx
    t0 = pd.Timestamp("2005-01-01"); ep = rx.drawdown_episodes(c, depth=-0.15)

    def thr(raw):  # 과거 분포 상위 20% 경계 (확장 윈도우, 오늘까지의 자료만)
        return raw.expanding(min_periods=252).quantile(0.8)

    def ev(nm, d):
        e = S.events(pct[(nm, d)] >= 0.8); return idx[e]

    def draw(panels, fname, title):
        n = len(panels) + 1
        fig, axes = plt.subplots(n, 1, figsize=(19, 3.3 * n), sharex=True, gridspec_kw={"height_ratios": [2.4] + [1.5] * len(panels)})
        cc = c.loc[t0:]; axes[0].semilogy(cc.index, cc, color="#1f3b73", lw=1.3); axes[0].set_ylabel("KOSPI"); axes[0].set_title(title, fontsize=15)
        for e in ep.itertuples():
            for a in axes: a.axvspan(max(e.peak, t0), e.trough, color=RED, alpha=0.10) if e.trough >= t0 else None
        for a, (lab, series, col, threshold, events) in zip(axes[1:], panels):
            s = series.loc[t0:]
            a.plot(s.index, s, color=col, lw=1.6, label="한국 10년 국채" if lab.startswith("금리 (%)") else None)
            if lab.startswith("금리 (%)"):
                a.plot(kr3.loc[t0:].index, kr3.loc[t0:], color=GREEN, lw=1.3, label="한국 3개월 은행간"); a.plot(tnx.loc[t0:].index, tnx.loc[t0:], color=GRAY, lw=1.3, label="미국 10년 국채"); a.legend(loc="upper right", fontsize=10.5)
            if threshold is not None:
                th = threshold.loc[t0:]; a.plot(th.index, th, color="k", ls="--", lw=1.2, label="상위 20% 경계(과거 분포 기준)")
                a.legend(loc="upper left", fontsize=10)
            if events is not None:
                ee = [d for d in events if d >= t0]
                if ee: a.scatter(ee, s.reindex(ee).values, marker="v", s=75, color=RED, edgecolors="k", linewidths=0.8, zorder=5)
            a.set_title(lab, loc="left"); a.grid(alpha=0.25)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2)); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=100); plt.close(fig)

    g60 = ratio.pct_change(60) * 100
    draw([("신용융자 ÷ 투자예탁금 (%) — '수준' 조건", ratio, GREEN, thr(ratio), None),
          ("신용융자 ÷ 투자예탁금의 60거래일 변화율 (%) — '증가' 조건 (둘 중 하나라도 상위 20%면 ▼)", g60, GREEN, thr(g60), ev("신용융자 과열(수준 또는 증가)", 60))],
         "new_1_신용융자과열.png", "새 지표 ① 신용융자 과열 (수준 또는 증가): 원본 값, 상위 20% 경계, 신호(▼)")
    d126 = kr10 - kr10.shift(126); sp126 = -(spread - spread.shift(126)); usd60 = (fx.pct_change(60) + (tnx - kr10) / 100 * 60 / 252) * 100
    draw([("금리 (%) — 한국 10년(월별, 2개월 지연 반영) / 한국 3개월 은행간 / 미국 10년", kr10, BLUE, None, None),
          ("한국 10년 금리의 126거래일 변화 (%p) — '한국 금리 상승'", d126, BLUE, thr(d126), ev("한국 금리 상승", 126)),
          ("한미 금리차 = 한국 10년 - 미국 10년 (%p)", spread, PURPLE, None, None),
          ("한미 금리차가 126거래일 동안 줄어든 폭 (%p) — '한미 금리차 축소'", sp126, PURPLE, thr(sp126), ev("한미 금리차 축소", 126)),
          ("원/달러 환율 (원)", fx, ORANGE, None, None),
          ("달러 보유 총수익 60거래일 (%) = 환율 상승률 + (미국-한국 10년 금리차)×60/252 — '환율·금리 결합'", usd60, ORANGE, thr(usd60), ev("환율·금리 결합(달러 총수익)", 60))],
         "new_2_금리와환율결합.png", "새 지표 ②③④ 한국 금리, 한미 금리차, 환율·금리 결합: 원본 값, 상위 20% 경계, 신호(▼)")
    print("완료")
    return None


if __name__ == "__main__":
    main()
