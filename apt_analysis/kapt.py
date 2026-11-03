"""K-apt(공동주택관리정보시스템) 세대수 수집 → data/households.csv

  export DATA_GO_KR_KEY=...        # 공공데이터포털 인증키 (코드에 넣지 말 것)
  python -m apt_analysis.kapt --data-dir <house_pjt>/web/public/data/gu

흐름
  1) AptListService4/getSigunguAptList4 로 시군구별 단지 목록(kaptCode, 단지명, 법정동)
  2) 실거래 단지(CSR)와 시군구+동+정규화 단지명으로 매칭
  3) 사전필터(회복률≥0, 대표평형≥40㎡)를 통과한 단지만
     AptBasisInfoServiceV4/getAphusBassInfoV4 로 기본정보(kaptdaCnt=세대수) 조회
     → 일일 호출 한도(1만) 절약. 응답은 data/cache/kapt 에 캐시해 재개 가능.
※ 두 오퍼레이션의 응답 필드명은 키 없이 실측하지 못했다. 파서는 JSON/XML 모두 받고
  필드가 다르면 FIELD_* 상수만 고치면 된다.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import requests

from . import metrics_cfg as M
from .common import Complex, anchor, load_all
from .scorecard import DONGTAN_DONGS, DONGTAN_SGG, in_universe

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LIST_URL = "https://apis.data.go.kr/1613000/AptListService4/getSigunguAptList4"
BASIS_URL = "https://apis.data.go.kr/1613000/AptBasisInfoServiceV4/getAphusBassInfoV4"

FIELD_CODE, FIELD_NAME, FIELD_BJD = "kaptCode", "kaptName", "bjdCode"
FIELD_DONG = ("as4", "as3")            # 법정동 이름 후보 필드
FIELD_HH = "kaptdaCnt"                 # 세대수
FIELD_USE = "kaptUsedate"              # 사용승인일 (YYYYMMDD)
FIELD_TYPE = ("codeAptNm", "codeSaleNm")   # 주택 유형/분양 구분
DAILY_LIMIT, MARGIN = 10_000, 200


class KaptError(RuntimeError):
    pass


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[()\[\]\s·.\-]", "", s)
    s = s.replace("제1차", "1차").replace("제2차", "2차").replace("제3차", "3차")
    return re.sub(r"(아파트|APT)$", "", s, flags=re.I)


def parse_response(text: str) -> tuple[str, int, list[dict]]:
    """(resultCode, totalCount, items). JSON/XML 모두 허용."""
    text = text.strip()
    if text.startswith("{"):
        j = json.loads(text)
        resp = j.get("response", j)
        code = str(resp.get("header", {}).get("resultCode", ""))
        body = resp.get("body", {}) or {}
        items = body.get("items") or []
        if isinstance(items, dict):
            items = items.get("item", [])
        if isinstance(items, dict):
            items = [items]
        return code, int(body.get("totalCount") or 0), list(items)
    root = ET.fromstring(text)
    code = (root.findtext(".//resultCode") or "").strip()
    total = int(root.findtext(".//totalCount") or 0)
    items = [{c.tag: (c.text or "").strip() for c in it} for it in root.iter("item")]
    return code, total, items


class Client:
    def __init__(self, key: str | None = None):
        raw = (key or os.getenv("DATA_GO_KR_KEY", "")).strip()
        if not raw:
            raise KaptError("환경변수 DATA_GO_KR_KEY 가 비어 있습니다.")
        self.key = unquote(raw)
        self.s = requests.Session()
        self.calls = 0

    def get(self, url: str, **params) -> tuple[int, list[dict]]:
        if self.calls >= DAILY_LIMIT - MARGIN:
            raise KaptError("일일 호출 한도 근접 — 내일 같은 명령으로 재개하세요 (캐시 사용).")
        params.update(serviceKey=self.key, _type="json")
        last = None
        for attempt in range(5):
            if attempt:
                time.sleep(2 ** (attempt - 1))
            try:
                r = self.s.get(url, params=params, timeout=20)
                self.calls += 1
                time.sleep(0.15)
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}"
                    continue
                code, total, items = parse_response(r.text)
                if code in ("00", "000", ""):
                    return total, items
                if code in ("22", "30", "31", "32"):
                    raise KaptError(f"API 오류 resultCode={code} (쿼터/키 문제)")
                last = f"resultCode={code}"
            except (requests.RequestException, ET.ParseError, ValueError) as e:
                last = str(e)
        raise KaptError(f"재시도 실패: {url.rsplit('/', 1)[-1]} {last}")


def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    v = fn()
    cache.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
    return v


def list_sigungu(cli: Client, sgg: str, cache: Path) -> list[dict]:
    def fetch():
        out, page = [], 1
        while True:
            total, items = cli.get(LIST_URL, sigunguCode=sgg, pageNo=page, numOfRows=1000)
            out += items
            if len(out) >= total or not items:
                return out
            page += 1
    return cached(cache, f"list_{sgg}", fetch)


def dong_of(it: dict) -> str:
    for f in FIELD_DONG:
        v = (it.get(f) or "").strip()
        if v.endswith(("동", "가", "리")):
            return v
    return ""


def match(c: Complex, entries: list[dict]) -> dict | None:
    """같은 동 안에서 정규화 단지명 일치 → 포함 → 유사도(≥0.8)."""
    pool = [e for e in entries if dong_of(e) == c.dong] or []
    if not pool:
        return None
    n = norm(c.name)
    for e in pool:
        if norm(e.get(FIELD_NAME)) == n:
            return e
    sub = [e for e in pool if n and (n in norm(e.get(FIELD_NAME)) or norm(e.get(FIELD_NAME)) in n)]
    if len(sub) == 1:
        return sub[0]
    best = max(pool, key=lambda e: difflib.SequenceMatcher(None, n, norm(e.get(FIELD_NAME))).ratio(),
               default=None)
    if best and difflib.SequenceMatcher(None, n, norm(best.get(FIELD_NAME))).ratio() >= 0.8:
        return best
    return None


def prefilter(c: Complex) -> bool:
    """API 호출 절약용 사전필터(세대수 무관한 조건만)."""
    p0, p1 = anchor(c, M.T0), anchor(c, M.T1)
    return bool(p0 and p1 and p1 / p0 - 1 >= M.MIN_RECOVERY and (c.rep_area or 0) >= M.MIN_REP_AREA)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--extra-dir", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=DATA / "households.csv")
    ap.add_argument("--cache", type=Path, default=DATA / "cache" / "kapt")
    a = ap.parse_args(argv)

    dirs = [a.data_dir, *a.extra_dir]
    if (DATA / "dongtan" / "gu").exists():
        dirs.append(DATA / "dongtan" / "gu")
    cs = [c for c in load_all(dirs)
          if in_universe(c.sgg) and not (c.sgg == DONGTAN_SGG and c.dong not in DONGTAN_DONGS)]
    cands = [c for c in cs if prefilter(c)]
    print(f"단지 {len(cs):,} → 사전필터 통과 {len(cands):,}")

    try:
        cli = Client()
    except KaptError as e:
        print(f"[중단] {e}", file=sys.stderr)
        return 2

    by_sgg: dict[str, list[dict]] = {}
    rows, unmatched = [], []
    try:
        for c in cands:
            if c.sgg not in by_sgg:
                by_sgg[c.sgg] = list_sigungu(cli, c.sgg, a.cache)
            e = match(c, by_sgg[c.sgg])
            if not e:
                unmatched.append(c)
                continue
            code = e[FIELD_CODE]
            info = cached(a.cache, f"basis_{code}", lambda: cli.get(BASIS_URL, kaptCode=code)[1])
            it = info[0] if info else {}
            typ = next((it.get(f) for f in FIELD_TYPE if it.get(f)), "")
            rows.append({"cid": c.cid, "kapt_code": code, "kapt_name": e.get(FIELD_NAME),
                         "households": it.get(FIELD_HH, ""), "use_date": it.get(FIELD_USE, ""),
                         "apt_type": typ})
    except KaptError as e:
        print(f"[중단] {e}", file=sys.stderr)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cid", "kapt_code", "kapt_name", "households", "use_date", "apt_type"]
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    um = a.out.with_name("households_unmatched.csv")
    with um.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cid", "gu", "dong", "name"])
        w.writerows([c.cid, c.gu_name, c.dong, c.name] for c in unmatched)
    print(f"매칭 {len(rows):,} · 미매칭 {len(unmatched):,} · API 호출 {cli.calls:,}회 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
