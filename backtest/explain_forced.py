"""반대매매 비중의 이상 움직임 원인 그림(한글): python3 backtest/explain_forced.py"""
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
RED, BLUE, GREEN, ORANGE, GRAY = "#e03131", "#1971c2", "#2b8a3e", "#f08c00", "#868e96"


def main():
    f = data.fetch_kofia_funds()
    unpaid, amt, ratio = f["unpaid"] / 100, f["forced_amount"] / 100, f["forced_ratio"]      # 억원
    ev = {"2023-10-18": ("영풍제지 하한가\n(10/18)", ORANGE), "2023-10-25": ("통계 집계 기준 변경 추정\n(대상금액→체결금액, 10/25)", RED), "2023-10-26": ("거래 재개(10/26)", GRAY)}
    fig, axes = plt.subplots(3, 2, figsize=(19, 12.5), gridspec_kw={"width_ratios": [1.1, 1.3]})
    for col, (t0, t1) in enumerate((("2023-09-20", "2023-12-15"), ("2021-01-01", "2026-11-15"))):
        for row, (s, lab, color) in enumerate(((unpaid, "위탁매매 미수금 잔고 (억원)", BLUE), (amt, "일별 반대매매 금액 (억원)", GREEN), (ratio, "미수금 대비 반대매매 비중 (%)", RED))):
            ax = axes[row, col]; x = s.loc[t0:t1]
            (ax.plot if col == 1 else ax.step)(*([x.index, x] if col == 1 else [x.index, x]), **({"color": color, "lw": 1.4} if col == 1 else {"color": color, "lw": 2, "where": "mid"}))
            ax.set_title(lab, loc="left"); ax.grid(alpha=0.25)
            if col == 0:
                ax.axvspan(pd.Timestamp("2023-10-18"), pd.Timestamp("2023-10-24 23:59"), color=ORANGE, alpha=0.15)
                for i_, (d, (txt, c_)) in enumerate(ev.items()):
                    ax.axvline(pd.Timestamp(d), color=c_, ls="--", lw=1.4)
                    if row == 0: ax.annotate(txt, (pd.Timestamp(d), x.max() * (1.0 - 0.17 * i_)), textcoords="offset points", xytext=(6, -2), fontsize=10, color=c_, fontweight="bold", va="top", bbox=dict(boxstyle="round", fc="white", ec=c_, alpha=0.9))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            else:
                ax.axvspan(pd.Timestamp("2023-10-18"), pd.Timestamp("2023-10-24"), color=ORANGE, alpha=0.4)
                ax.axvline(pd.Timestamp("2023-10-25"), color=RED, ls="--", lw=1.4)
                ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].text(0.02, 0.38, "사건 전 미수금 ≈ 5,300억\n사건 후 ≈ 9,700억 (+약 4,300억)\n= 키움증권 영풍제지 미수채권\n  남은 금액 4,333억(11/6 공시)과 일치", transform=axes[0, 0].transAxes, fontsize=10.5, va="center", bbox=dict(boxstyle="round", fc="white", ec=BLUE))
    axes[1, 0].text(0.40, 0.52, "10/18~24: 거래정지로 체결되지 못한\n반대매매 '대상금액'이 매일 겹쳐 집계\n(실제 체결은 평소 수준 ≈ 544억, 10/24)", transform=axes[1, 0].transAxes, fontsize=10.5, va="center", bbox=dict(boxstyle="round", fc="white", ec=GREEN))
    axes[1, 0].text(0.57, 0.12, "10/25 이후: 평소의 약 1/8 수준\n(체결금액 기준으로 집계)", transform=axes[1, 0].transAxes, fontsize=10.5, va="center", bbox=dict(boxstyle="round", fc="white", ec=RED))
    axes[2, 0].text(0.02, 0.55, "평소 약 9~10% → 10/18~24 최대 69%\n→ 10/25 이후 약 0.3~1%", transform=axes[2, 0].transAxes, fontsize=10.5, va="center", bbox=dict(boxstyle="round", fc="white", ec=RED))
    axes[2, 1].text(0.03, 0.75, "집계 기준 변경으로 2023-10-25 전후는\n같은 잣대로 비교할 수 없음\n(미수금 잔고도 +4,300억 높아짐)", transform=axes[2, 1].transAxes, fontsize=11, va="center", bbox=dict(boxstyle="round", fc="white", ec=RED))
    fig.suptitle("반대매매 비중이 69%로 폭등했다가 0.6~1%로 내려앉은 이유 — 영풍제지 사태와 통계 집계 방식 변경 (왼쪽=2023-09~12 확대, 오른쪽=2021~현재)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "forced_1_원인.png", dpi=105); plt.close(fig)
    print("완료")


if __name__ == "__main__":
    main()
