"""'폭락 시점' 평가: 고점 전후 ±b% 구간 안에 신호가 있으면 적중.

- 폭락 고점: 종가 기준 지그재그(반전폭 dev)로 찾은 고점 중, 이후 저점까지 dev(15%) 이상 하락한 것 (국지적 고점 포함)
- 구간: 시작 = 고점으로 마지막 b% 상승을 시작한 날(고점 직전에 종가가 peak/(1+b) 이하였던 마지막 날), 끝 = 고점 후 종가가 peak×(1-b) 이하로 처음 내려간 날
- 신호 이벤트: 조건이 새로 참이 된 날(상승 에지), 이벤트 간 최소 cool 거래일
"""
import numpy as np
import pandas as pd


def zigzag_tops(close: pd.Series, dev: float = 0.15):
    c = close.values; n = len(c)
    tops, bots = [], []
    mode = 0; ext_i = 0
    for i in range(1, n):
        if mode == 0:
            if c[i] >= c[ext_i] * (1 + dev): mode = 1; ext_i = i; bots.append(0)
            elif c[i] <= c[ext_i] * (1 - dev): mode = -1; ext_i = i; tops.append(0)
            continue
        if mode == 1:                      # 상승 중: 고점 갱신
            if c[i] > c[ext_i]: ext_i = i
            elif c[i] <= c[ext_i] * (1 - dev): tops.append(ext_i); mode = -1; ext_i = i
        else:                              # 하락 중: 저점 갱신
            if c[i] < c[ext_i]: ext_i = i
            elif c[i] >= c[ext_i] * (1 + dev): bots.append(ext_i); mode = 1; ext_i = i
    return [t for t in tops if t > 0]


def make_windows(close: pd.Series, tops, band: float = 0.05):
    c = close.values; n = len(c); out = []
    for t in tops:
        s = t
        while s > 0 and c[s] > c[t] / (1 + band): s -= 1       # 고점에서 b% 아래였던 마지막 날까지 후진
        e = t
        while e < n - 1 and c[e] > c[t] * (1 - band): e += 1   # 고점 후 b% 하락한 첫날
        out.append((s, t, e))
    return out


def events_from_cond(cond, start_i: int, cool: int = 20):
    cond = np.asarray(cond, bool).copy(); cond[:start_i] = False
    rising = np.flatnonzero(cond[1:] & ~cond[:-1]) + 1
    ev, last = [], -10**9
    for i in rising:
        if i - last >= cool:
            ev.append(i); last = i
    return np.array(ev, int)


def score_events(ev, windows):
    """적중 구간 수, 구간 안 이벤트 수, 구간별 첫 적중(고점 대비 일수, 음수=고점 전)"""
    hit, first, inside = 0, [], 0
    inwin = np.zeros(len(ev), bool)
    for (s, t, e) in windows:
        m = (ev >= s) & (ev <= e)
        inwin |= m
        if m.any():
            hit += 1; first.append(int(ev[m][0] - t))
        else:
            first.append(None)
    return hit, int(inwin.sum()), first
