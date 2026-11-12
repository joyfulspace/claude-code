"""scorecard_all.csv → 열람용 한글 CSV/XLSX (data/scorecard_all_view.*). 사용: python -m apt_analysis.export_view"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    a = pd.read_csv(DATA / "scorecard_all.csv", encoding="utf-8-sig")
    a = a.sort_values(["pass_all", "score100"], ascending=[False, False], na_position="last").reset_index(drop=True)
    ox = {True: "O", False: "X"}
    o = pd.DataFrame({
        "순위": a.index + 1, "통과": a.pass_all.map({True: "통과", False: ""}), "구": a.gu, "동": a.dong, "단지명": a.name,
        "점수(100)": a.score100, "점수(5)": a.score, "세대수": a.households, "준공연도": a.build_year,
        "대표평형(㎡)": a.rep_area_m2, "84㎡환산(억)": (a.price84_filled / 10000).round(2),
        "가격기준": a.price84_src.fillna("").replace({"2025-09": "2025-09"}),
        "회복률": a.recovery.round(4), "고점대비": a.from_peak.round(4), "을지로거리(km)": a.dist_km.round(2),
        "12개월월평균거래": a.vol_12m.round(2),
        "입지": a.s_loc, "하락방어": a.s_defense, "생활수준": a.s_income, "유동성": a.s_liquidity,
        "연식": a.s_age, "예산": a.s_budget, "거리": a.s_dist,
        "회복률필터": a.f_recovery.map(ox), "면적필터": a.f_area.map(ox), "도시형필터": a.f_urban.map(ox),
        "세대수필터": a.f_households.map(ox), "거래량필터": a.f_turnover.map(ox),
        "K-apt매칭": a.kapt_matched.map(ox), "좌표추정": a.coord_estimated.map({True: "추정", False: ""}), "단지ID": a.cid})
    o.to_csv(DATA / "scorecard_all_view.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(DATA / "scorecard_all_view.xlsx", engine="openpyxl") as w:
        o.to_excel(w, index=False, sheet_name="스코어카드_전체")
        ws = w.sheets["스코어카드_전체"]
        ws.freeze_panes = "F2"
        ws.auto_filter.ref = ws.dimensions
        for col, wd in {"A": 6, "B": 6, "C": 8, "D": 10, "E": 28}.items():
            ws.column_dimensions[col].width = wd


if __name__ == "__main__":
    main()
