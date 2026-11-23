"""지표별 파라미터 최적화 → 최적값으로 지표 비교.   python3 backtest/run_param_opt.py

[1] '하락을 잘 피했다'의 정의 (점수 = 조화평균)
    R  = 큰 하락(고점→저점 -15%↑) 구간들에서 전체 낙폭의 평균 몇 %를 피했나 (매도 못 하면 0%)  ← 많이 피할수록 ↑
    Pk = 매도 신호 뒤 120거래일 안에 체결가보다 -10% 더 떨어진 비율 (유효율; 표본이 적으면 기준값 쪽으로 보정) ← 헛발질이 적을수록 ↑
    점수 = 2·R·Pk / (R + Pk)   → 둘 다 높아야 높은 점수
[2] 규칙 = (고점 대비 D% 이하) AND (지표가 임계 백분위 θ 이상/이하) , 지표 창 길이 w 도 함께 탐색
    검증: 기간 교차검증(~2012로 찾아 2013~에 적용, 반대도) + 가짜 지표(무작위로 시간축을 밀어 둔 지표) 최적화로 만든 '우연의 기준선'
[3] 최적값으로 지표끼리 비교
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data as data_mod  # noqa: E402
import risk_exit as rx  # noqa: E402
from run_sell_lab import load_all  # noqa: E402

OUT = Path(__file__).parent / "results"
FWD, GOODDROP, COOL = 120, -0.10, 120
START = pd.Timestamp("2006-01-02")
SPLIT = pd.Timestamp("2013-01-01")
D_GRID = [-5, -6, -8, -10, -12, -15]   # '하락 시작'의 최소 기준을 -5%로 둔다(-3~-4% 조정은 제외)
TH_GRID = [20, 30, 40, 50, 60, 70, 80, 90]
K_SHRINK = 5


def P(s):  # 오늘까지의 분포 대비 백분위 (최소 1년)
    return rx.pct_rank(s, min_periods=252)


def build_specs(k, ser, funds):
    c, idx = k["close"], k.index
    ret = c.pct_change()
    lag = lambda s: s.reindex(idx).ffill().shift(1)
    spx, vix, fx, tnx = lag(ser["SPX"]), lag(ser["VIX"]), lag(ser["USDKRW"]), lag(ser["TNX"])
    kq = ser["KOSDAQ"].reindex(idx).ffill()
    us, fxr, vixc = spx.pct_change(), fx.pct_change(), vix.pct_change()
    f = funds.reindex(idx).ffill().shift(1)
    down, up = us < 0, us > 0

    def beta(m, w):
        return (ret * us).where(m).rolling(w, min_periods=30).mean() / (us ** 2).where(m).rolling(w, min_periods=30).mean()
    ratio = f["margin"] / f["deposit"]
    kr10 = data_mod.monthly_to_daily(ser["KR10Y"], idx) if "KR10Y" in ser else None
    kr3 = data_mod.monthly_to_daily(ser["KR3M"], idx) if "KR3M" in ser else None

    def pct_named(x):
        x = x.copy(); x.name = "__pct__"; return x

    forced_adj_series = data_mod.forced_ratio_adjusted(funds)["adj"].reindex(idx).ffill().shift(1)

    def forced_adj(w):
        """반대매매 비중(체결금액 기준으로 환산·복원한 보정 시계열)의 w일 평균 → 백분위"""
        return forced_adj_series.rolling(w).mean()

    def credit_heat(w):
        """신용융자/예탁금이 높거나(수준) 최근 증가율이 과도하면(증가) 울림 = 두 백분위의 최대값(OR)"""
        lvl = P(ratio.rolling(5).mean()); gr = P(ratio.pct_change(w))
        return pct_named(pd.concat([lvl, gr], axis=1).max(axis=1, skipna=False))
    S = {
        ("수급", "신용융자/예탁금"): ("평활일수", [1, 5, 20, 60], 1, lambda w: ratio.rolling(w).mean()),
        ("수급", "예탁금 감소"): ("변화창", [20, 40, 60, 120], 60, lambda w: -f["deposit"].pct_change(w)),
        ("수급", "반대매매 비중"): ("평균일수", [5, 20, 60, 120], 20, forced_adj),
        ("수급", "신용융자 증가"): ("변화창", [20, 40, 60, 120], 60, lambda w: f["margin"].pct_change(w)),
        ("수급", "원화 약세"): ("변화창", [20, 40, 60, 120], 60, lambda w: fx.pct_change(w)),
        ("시장민감도", "한미 상관"): ("창", [20, 40, 60, 120], 60, lambda w: ret.rolling(w).corr(us)),
        ("시장민감도", "미국 베타"): ("창", [20, 40, 60, 120], 60, lambda w: ret.rolling(w).cov(us) / us.rolling(w).var()),
        ("시장민감도", "하락비대칭 베타"): ("창", [60, 120, 250], 120, lambda w: beta(down, w) - beta(up, w)),
        ("시장민감도", "VIX 변화 민감도"): ("창", [20, 40, 60, 120], 60, lambda w: -ret.rolling(w).corr(vixc)),
        ("시장민감도", "환율 민감도"): ("창", [20, 40, 60, 120], 60, lambda w: -ret.rolling(w).corr(fxr)),
        ("시장민감도", "미국금리 상승"): ("변화창", [20, 40, 60, 120], 60, lambda w: tnx.diff(w)),
        ("심리", "VIX 수준"): ("평활일수", [1, 5, 10, 20], 1, lambda w: vix.rolling(w).mean()),
        ("심리", "VIX 급등"): ("변화창", [1, 3, 5, 10, 20], 5, lambda w: vix.pct_change(w)),
        ("심리", "코스닥 상대약세"): ("변화창", [10, 20, 40, 60], 20, lambda w: -(kq / c).pct_change(w)),
        ("심리", "하락일 비중"): ("창", [10, 20, 40, 60], 20, lambda w: (ret < 0).astype(float).rolling(w).mean()),
        ("심리", "RSI"): ("RSI기간", [7, 14, 28], 14, lambda w: rx.rsi(c, w)),
    }
    S[("수급", "신용융자 과열(수준 또는 증가)")] = ("증가창", [20, 40, 60, 120], 60, credit_heat)
    if kr10 is not None:
        spread = kr10 - tnx
        S[("금리·환율", "한국 금리 상승")] = ("변화창(거래일)", [63, 126, 252], 126, lambda w: kr10 - kr10.shift(w))
        S[("금리·환율", "한미 금리차 축소")] = ("변화창(거래일)", [63, 126, 252], 126, lambda w: -(spread - spread.shift(w)))
        S[("금리·환율", "환율·금리 결합(달러 총수익)")] = ("변화창(거래일)", [20, 60, 120], 60, lambda w: fx.pct_change(w) + (tnx - kr10) / 100 * w / 252)
    return S


def pct_of(fn, w):
    s = fn(w)
    return s.values if getattr(s, "name", None) == "__pct__" else P(s).values


def ensure_cache(specs, cache_path):
    """백분위 캐시 갱신: 바뀐 지표(반대매매 비중)는 다시 계산하고 새 지표를 추가."""
    pct = pd.read_pickle(cache_path) if Path(cache_path).exists() else {}
    for key in [k_ for k_ in pct if k_[0] in ("반대매매 비중",)]:
        del pct[key]
    for (g, nm), (pn, wins, dflt, fn) in specs.items():
        for w in wins:
            if (nm, w) not in pct:
                pct[(nm, w)] = pct_of(fn, w)
    pd.to_pickle(pct, cache_path)
    return pct


class Sim:
    def __init__(self, k):
        c, o = k["close"].values, k["open"].values
        self.idx, self.c, self.o, self.n = k.index, c, o, len(c)
        s = k["close"]
        self.dd = (s / s.rolling(252, min_periods=60).max() - 1).values
        self.start_i = int(self.idx.searchsorted(START))
        n = self.n
        self.complete = np.zeros(n, bool); self.valid = np.zeros(n, bool); self.ret_end = np.full(n, np.nan)
        for i in range(n - 1):
            if i + 1 + FWD < n:
                Pp = o[i + 1]; w = c[i + 1:i + 1 + FWD]
                self.complete[i] = True; self.valid[i] = (w.min() / Pp - 1) <= GOODDROP; self.ret_end[i] = c[i + FWD] / Pp - 1
        ep = rx.drawdown_episodes(s.loc[START:], depth=-0.15)
        self.ep = [(self.idx.get_loc(r.peak), self.idx.get_loc(r.trough), r.peak, r.depth) for r in ep.itertuples()]
        self.periods = {"full": (START, pd.Timestamp("2100-01-01")), "P1": (START, SPLIT - pd.Timedelta(days=1)), "P2": (SPLIT, pd.Timestamp("2100-01-01"))}
        self.p0 = 0.35

    def run(self, cond):
        n, idx = self.n, self.idx
        cond = cond.copy(); cond[:self.start_i] = False
        rising = np.flatnonzero(cond[1:] & ~cond[:-1]) + 1
        ev, last = [], -10**9
        for i in rising:
            if i - last >= COOL:
                ev.append(i); last = i
        ev = np.array(ev, int)
        # 구간별 회피비율
        av = []
        for pk, tr, pkd, depth in self.ep:
            seg = cond[pk:tr + 1]
            if seg.any():
                i0 = pk + int(np.argmax(seg)); Pp = self.o[i0 + 1] if i0 + 1 < n else self.c[i0]
                av.append((pkd, max((Pp - self.c[tr]) / (self.c[pk] - self.c[tr]), 0.0), (idx[i0] - pkd).days))
            else:
                av.append((pkd, 0.0, np.nan))
        res = {}
        for name, (a, b) in self.periods.items():
            a_ = [x[1] for x in av if a <= x[0] <= b]
            R = float(np.mean(a_)) if a_ else np.nan
            e = [i for i in ev if a <= idx[i] <= b and self.complete[i]]
            nv = int(sum(self.valid[i] for i in e)); ne = len(e)
            Pk = (nv + K_SHRINK * self.p0) / (ne + K_SHRINK)
            sc = 2 * R * Pk / (R + Pk) if (R == R and R + Pk > 0) else np.nan
            net = float(-sum(self.ret_end[i] for i in e))  # 팔지 않고 120일 보유했을 때 대비 이득의 합(+이면 팔길 잘함)
            res[name] = (sc, R, Pk, ne, nv, int(sum(x >= .25 for x in a_)), len(a_), len(e) and float(nv / ne), net)
        return res


def main():
    pd.options.display.float_format = "{:.3f}".format
    OUT.mkdir(exist_ok=True)
    k, ser, funds = load_all()
    sim = Sim(k)
    S = build_specs(k, ser, funds)
    cache = OUT / "param_opt_pct.pkl"
    pct = pd.read_pickle(cache) if cache.exists() else {}
    pct = ensure_cache(S, cache)
    dd = sim.dd

    # 기준 유효율(prior): 기본 규칙 -8%
    base8 = sim.run(dd <= -0.08)["full"]
    sim.p0 = base8[7] if base8[7] == base8[7] else 0.35
    print(f"평가 구간 {START.date()}~ | 큰 하락 {len(sim.ep)}개 | 기준 규칙(고점 -8%) 유효율 {sim.p0:.2f} (표본 보정의 기준값)\n")

    import warnings
    warnings.simplefilter("ignore")
    nD, nT = len(D_GRID), len(TH_GRID)

    def grid_scores(arrs):
        """arrs: 지표 백분위 배열(창 후보들) → 점수 배열 (기간3, 창, 방향2, D, θ)"""
        out = np.full((3, len(arrs), 2, nD, nT), np.nan)
        for wi, arr in enumerate(arrs):
            for di, D in enumerate(D_GRID):
                base_c = dd <= D / 100
                for ti, th in enumerate(TH_GRID):
                    for hi, direction in enumerate((arr >= th / 100, arr <= 1 - th / 100)):
                        r = sim.run(base_c & direction)
                        out[:, wi, hi, di, ti] = (r["full"][0], r["P1"][0], r["P2"][0])
        return out

    def smooth(a):  # 마지막 두 축(D, θ) 3x3 이웃 평균 → 한 점이 튀는 값을 덜 고르게
        pad = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], constant_values=np.nan)
        st = np.stack([pad[..., i:i + nD, j:j + nT] for i in range(3) for j in range(3)])
        return np.nanmean(st, axis=0)

    def pick(sc):
        """sc: (3, 창, 2, D, θ). 반환: 전체 최적 인덱스, 검증점수(교차), 각 fold 점수"""
        sm = smooth(sc)
        def am(x):
            return np.unravel_index(np.argmax(np.nan_to_num(x, nan=-1)), x.shape)
        f = am(sm[0]); a1 = am(sm[1]); a2 = am(sm[2])
        t_a = sc[(2,) + a1]; t_b = sc[(1,) + a2]
        return f, np.nanmean([t_a, t_b]), t_a, t_b

    grp = {nm: g for (g, nm) in S}; grp["기준(지표 없음)"] = "기준"
    summ, chosen = [], {}
    for (g, nm), (pn, wins, dflt, fn) in S.items():
        arrs = [pct[(nm, w)] for w in wins]
        sc = grid_scores(arrs)
        f, oos, ta, tb = pick(sc)
        wi, hi, di, ti = f
        D, th, direction = D_GRID[di], TH_GRID[ti], ("높을때", "낮을때")[hi]
        arr = arrs[wi]
        r = sim.run((dd <= D / 100) & ((arr >= th / 100) if hi == 0 else (arr <= 1 - th / 100)))["full"]
        dflt_sc = sc[(0, wins.index(dflt), 0, D_GRID.index(-8), TH_GRID.index(70))]
        chosen[nm] = sc
        summ.append({"그룹": g, "지표": nm, "기본값점수": dflt_sc, "파라미터명": pn, "최적창": wins[wi], "최적D": D, "최적θ": th, "방향": direction,
                     "회피비율": r[1], "유효율": r[7], "발동": r[3], "회피25%↑": f"{r[5]}/{r[6]}", "최적점수(전체)": r[0], "검증점수(교차)": oos, "검증A": ta, "검증B": tb, "120일순절약합": r[8]})
    # 기준 규칙: D 만 탐색
    bsc = np.array([[sim.run(dd <= D / 100)[p][0] for D in D_GRID] for p in ("full", "P1", "P2")])
    smb = np.array([[np.nanmean(row[max(i - 1, 0):i + 2]) for i in range(nD)] for row in bsc])
    bf, b1, b2 = (int(np.argmax(np.nan_to_num(smb[j], nan=-1))) for j in range(3))
    rb = sim.run(dd <= D_GRID[bf] / 100)["full"]
    summ.append({"그룹": "기준", "지표": "기준(지표 없음)", "기본값점수": bsc[0][D_GRID.index(-8)], "파라미터명": "-", "최적창": 0, "최적D": D_GRID[bf], "최적θ": 0, "방향": "-",
                 "회피비율": rb[1], "유효율": rb[7], "발동": rb[3], "회피25%↑": f"{rb[5]}/{rb[6]}", "최적점수(전체)": rb[0],
                 "검증점수(교차)": np.nanmean([bsc[2][b1], bsc[1][b2]]), "검증A": bsc[2][b1], "검증B": bsc[1][b2], "120일순절약합": rb[8]})
    SM = pd.DataFrame(summ)

    # 우연의 기준선: 가짜 지표(백분위를 무작위로 시간 이동) 4개 중 최적 → 같은 절차(이웃 평균 선택 + 교차검증)
    rng = np.random.default_rng(0); n = sim.n
    keys = [(nm, dflt) for (g, nm), (pn, w_, dflt, fn) in S.items()]
    nf, no = [], []
    for _ in range(150):
        arrs = [np.roll(pct[keys[rng.integers(len(keys))]], int(rng.integers(500, n - 500))) for _ in range(4)]
        sc = grid_scores(arrs)
        f, oos, ta, tb = pick(sc)
        nf.append(sc[(0,) + f]); no.append(oos)
    nf, no = np.array(nf), np.array(no)
    SM["전체 우연기준(95%)"] = np.nanpercentile(nf, 95); SM["검증 우연기준(95%)"] = np.nanpercentile(no, 95)
    SM = SM.sort_values("검증점수(교차)", ascending=False)
    SM.to_csv(OUT / "param_opt_summary.csv", index=False, encoding="utf-8-sig")
    pd.to_pickle({"null_full": nf, "null_oos": no, "p0": sim.p0, "scores": chosen, "bsc": bsc, "S": {nm: (pn, wins, dflt) for (g, nm), (pn, wins, dflt, fn) in S.items()}}, OUT / "param_opt_null.pkl")
    print(f"[우연의 기준선] 가짜 지표 150회 (창 4개 탐색과 같은 자유도, 같은 절차):\n   전체구간 최고점수 평균 {np.nanmean(nf):.3f}, 95% {np.nanpercentile(nf, 95):.3f} | 검증점수 평균 {np.nanmean(no):.3f}, 95% {np.nanpercentile(no, 95):.3f}\n")
    print("[3] 지표 비교 (최적값 기준). 최적점수=전체 구간에서 맞춘 점수(과적합 포함) / 검증점수=한 구간에서 찾은 값을 다른 구간에 적용한 평균")
    print(SM.drop(columns=["검증A", "검증B", "파라미터명", "전체 우연기준(95%)", "검증 우연기준(95%)"]).to_string(index=False))


if __name__ == "__main__":
    main()
