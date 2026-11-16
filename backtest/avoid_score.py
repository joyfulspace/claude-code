"""'하락을 잘 피했다' 점수 (확정안, 재진입 없음).

신호 = 지표 단독(또는 임의의 bool 시계열). 신호일 종가에 확인 → 다음날 시가에 전량 매도했다고 가정. 120거래일 측정 창. 한 번 울리면 120일간은 다음 신호를 세지 않음(중복 방지).
· 피한 손실(큰 하락 구간별): 구간(고점 60일 전~저점) 안에서 처음 울린 신호(한 신호는 고점에 가장 가까운 한 구간에만 인정)의 매도가 P 에 대해
      금액 B = (P - 저점)/P      (매도한 돈의 몇 %가 더 빠지는 걸 피했나)
      비율 A = (P - 저점)/(고점 - 저점)   (구간 낙폭의 몇 %를 피했나; 못 팔면 0, 100% 초과/음수는 0~1로 자름)
· 기회비용(모든 신호): 매도 후 120거래일 안에 매도가 대비 최고가까지 오른 폭 m = max(0, 최고/P - 1)
· 점수  금액 = 평균 B - λ × 평균 m           비율 = 평균 A - λ × (평균 m / 평균 큰 하락 낙폭)
"""
import numpy as np
import pandas as pd

import risk_exit as rx

FWD, COOL, PRE = 120, 120, 60
START = pd.Timestamp("2006-01-02")
LAMBDAS = (0.25, 0.5, 1.0)


class Scorer:
    def __init__(self, k, start=START, depth=-0.15):
        s = k["close"]
        self.idx, self.c, self.o, self.n = k.index, s.values, k["open"].values, len(s)
        self.start_i = int(self.idx.searchsorted(start))
        ep = rx.drawdown_episodes(s.loc[start:], depth=depth)
        self.ep = [(self.idx.get_loc(r.peak), self.idx.get_loc(r.trough), r.peak, r.depth) for r in ep.itertuples()]
        self.mean_depth = float(-np.mean([e[3] for e in self.ep]))
        n = self.n
        self.m = np.full(n, np.nan); self.a = np.full(n, np.nan)   # 신호일 i 기준 놓친 상승 / 이후 최저까지 하락폭
        self.wlen = np.zeros(n, int)
        for i in range(n - 1):
            P = self.o[i + 1]; w = self.c[i + 1:i + 1 + FWD]
            if len(w) >= 20:
                self.m[i] = max(0.0, w.max() / P - 1); self.a[i] = max(0.0, 1 - w.min() / P); self.wlen[i] = len(w)

    def events(self, cond):
        cond = np.asarray(cond, bool).copy(); cond[:self.start_i] = False
        rising = np.flatnonzero(cond[1:] & ~cond[:-1]) + 1
        ev, last = [], -10**9
        for i in rising:
            if i - last >= COOL:
                ev.append(i); last = i
        return np.array([i for i in ev if self.wlen[i] > 0], int)

    def score(self, cond):
        """신호 = events(): 120일 간격으로 센 신호들. 한 신호는 자신이 속한 구간 중 고점에 가장 가까운 한 구간에만 공을 인정."""
        cond = np.asarray(cond, bool).copy(); cond[:self.start_i] = False
        ev = self.events(cond)
        owner = {}
        for i in ev:
            cands = [(abs(i - pk), j) for j, (pk, tr, _, _) in enumerate(self.ep) if max(pk - PRE, self.start_i) <= i <= tr]
            if cands:
                owner[i] = min(cands)[1]
        B, A, first = [], [], []
        for j, (pk, tr, pkd, depth) in enumerate(self.ep):
            mine = [i for i in ev if owner.get(i) == j]
            if mine:
                i0 = mine[0]; P = self.o[i0 + 1] if i0 + 1 < self.n else self.c[i0]
                B.append(max((P - self.c[tr]) / P, 0.0)); A.append(float(np.clip((P - self.c[tr]) / (self.c[pk] - self.c[tr]), 0, 1)))
                first.append((self.idx[i0] - pkd).days)
            else:
                B.append(0.0); A.append(0.0)
        m = float(np.mean(self.m[ev])) if len(ev) else 0.0
        out = {"신호": len(ev), "평균 피한손실(금액)": float(np.mean(B)), "평균 회피비율": float(np.mean(A)), "평균 놓친상승": m,
               "큰하락 중 신호": int(sum(b > 0 for b in B)), "발동(고점대비 일, 중앙)": float(np.median(first)) if first else np.nan}
        for lam in LAMBDAS:
            out[f"금액 λ={lam}"] = out["평균 피한손실(금액)"] - lam * m
            out[f"비율 λ={lam}"] = out["평균 회피비율"] - lam * m / self.mean_depth
        return out


def detail(self, cond):
    """최적화·교차검증용 세부값: 구간별 피한 손실 B_j(금액형), 신호 인덱스, 신호별 놓친 상승 m."""
    cond = np.asarray(cond, bool).copy(); cond[:self.start_i] = False
    ev = self.events(cond)
    owner = {}
    for i in ev:
        cands = [(abs(i - pk), j) for j, (pk, tr, _, _) in enumerate(self.ep) if max(pk - PRE, self.start_i) <= i <= tr]
        if cands:
            owner[i] = min(cands)[1]
    Bj = np.zeros(len(self.ep))
    for j, (pk, tr, pkd, depth) in enumerate(self.ep):
        mine = [i for i in ev if owner.get(i) == j]
        if mine:
            i0 = mine[0]; P = self.o[i0 + 1] if i0 + 1 < self.n else self.c[i0]
            Bj[j] = max((P - self.c[tr]) / P, 0.0)
    return {"B": Bj, "ev": ev, "m": self.m[ev] if len(ev) else np.array([]), "dates": self.idx[ev] if len(ev) else pd.DatetimeIndex([])}


Scorer.detail = detail
