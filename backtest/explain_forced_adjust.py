"""반대매매 비중 보정 결과 그림·CSV(한글): python3 backtest/explain_forced_adjust.py"""
import sys
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402

OUT = Path(__file__).parent / "results"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, ORANGE = "#1971c2", "#e03131", "#f08c00"


def main():
    f = pd.read_csv(data.DATA_DIR / "KOFIA_funds.csv", index_col=0, parse_dates=True)
    o = data.forced_ratio_adjusted(f, verbose=True); k = o.attrs["k"]
    o.rename(columns={"raw": "원자료(%)", "adj": "보정(%)", "basis": "원자료의 집계기준", "unpaid_adj": "보정 미수금(억원)"}).to_csv(OUT / "forced_ratio_adjusted.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(19, 9.5), sharex="col", gridspec_kw={"width_ratios": [1.5, 1]})
    for col, (t0, t1) in enumerate((("2006-04-14", "2026-11-15"), ("2023-09-01", "2024-02-28"))):
        a, b = axes[0, col], axes[1, col]
        r, d = o["raw"].loc[t0:t1], o["adj"].loc[t0:t1]
        a.plot(r.index, r, color=BLUE, lw=1.1); a.set_title("원자료: 반대매매 비중 (%)", loc="left"); a.grid(alpha=0.25)
        b.plot(d.index, d, color=RED, lw=1.1); b.set_title("보정 후: 체결금액 기준 수준으로 환산·복원 (%)", loc="left"); b.grid(alpha=0.25)
        for ax in (a, b):
            ax.axvspan(pd.Timestamp("2023-10-18"), pd.Timestamp("2023-10-24"), color=ORANGE, alpha=0.35)
            ax.axvline(pd.Timestamp("2023-10-25"), color="k", ls="--", lw=1.0)
        if col == 0:
            b.xaxis.set_major_locator(mdates.YearLocator(2)); b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            a.annotate("2023-10-18~24 영풍제지 중복집계\n(최대 69%)", (pd.Timestamp("2023-10-20"), 66), xytext=(-250, -5), textcoords="offset points", arrowprops=dict(arrowstyle="->"), fontsize=11, fontweight="bold")
            a.annotate("2023-10-25 집계기준 변경\n(대상금액 → 체결금액)", (pd.Timestamp("2023-10-25"), 5), xytext=(40, 45), textcoords="offset points", arrowprops=dict(arrowstyle="->"), fontsize=11, fontweight="bold", color="k")
            b.text(0.02, 0.8, f"환산계수 k = {k:.3f}\n(사건 전 6개월 중앙값 {o.attrs['pre_med']:.1f}%\n → 사건 후 6개월 {o.attrs['post_med']:.2f}%)\n과거치 × k, 분모에서 영풍제지 미수채권 4,333억 제외", transform=b.transAxes, fontsize=10.5, va="top", bbox=dict(boxstyle="round", fc="white", ec=RED))
        else:
            b.xaxis.set_major_locator(mdates.MonthLocator()); b.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
            b.text(0.02, 0.92, "폭등 5일이 주변 수준으로 복원\n전후가 끊기지 않고 이어짐", transform=b.transAxes, fontsize=10.5, va="top", bbox=dict(boxstyle="round", fc="white", ec=RED))
    fig.suptitle("반대매매 비중 보정: 영풍제지 중복집계 제거 + 집계기준 변경(대상→체결)에 맞춰 과거치 환산 (주황 음영 = 10/18~24, 점선 = 10/25)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "forced_2_보정결과.png", dpi=105); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
