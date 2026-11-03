"""화성시 동탄(41590) 실거래 수집 → CSR 지수 (house_pjt 와 동일 방식).

house_pjt 의 pipeline(collect → build → export)을 그대로 재사용한다.
설정(저장 위치·대상 시군구)만 import 전에 덮어쓰고, 결과는 data/dongtan/ 에 둔다.

  export DATA_GO_KR_KEY=...
  python -m apt_analysis.dongtan --house-pjt <clone 경로> [--from 202001] [--skip-collect]

산출: data/dongtan/gu/41590.json (+ .areas.json) — scorecard 가 자동 포함.
※ 키는 '국토교통부 아파트 매매 실거래가 상세 자료' 활용신청이 승인된 키여야 한다.
※ 좌표(지오코딩)는 카카오 키가 필요해 건너뛴다 → 동탄 단지는 대략 중심 좌표로 거리 점수 계산.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dongtan"
SGG = "41590"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--house-pjt", type=Path, required=True, help="woongcomi/house_pjt 클론 경로")
    ap.add_argument("--from", dest="ym_from", default="202001",
                    help="수집 시작월 (회복률은 2021.1~ 필요; 기본 2020-01)")
    ap.add_argument("--to", dest="ym_to", default=None)
    ap.add_argument("--skip-collect", action="store_true", help="저장된 원시 XML 로 재적재")
    a = ap.parse_args(argv)

    if not os.getenv("DATA_GO_KR_KEY", "").strip():
        print("[중단] 환경변수 DATA_GO_KR_KEY 가 비어 있습니다.", file=sys.stderr)
        return 2
    pipe = a.house_pjt / "pipeline"
    if not (pipe / "collect.py").exists():
        print(f"[중단] {pipe}/collect.py 없음", file=sys.stderr)
        return 2

    sys.path.insert(0, str(pipe))
    import config as C                                   # noqa: E402  (다른 모듈보다 먼저)
    C.ALL_GU = {SGG: "화성시"}
    C.SIDO_OF = {SGG: "화성"}
    C.DB_PATH = OUT / "house.db"
    C.RAW_DIR = OUT / "raw"
    C.EXPORT_DIR = OUT / "export"
    OUT.mkdir(parents=True, exist_ok=True)

    import build                                          # noqa: E402
    import collect                                        # noqa: E402
    import export                                         # noqa: E402

    args = ["--lawd", SGG, "--from", a.ym_from]
    if a.ym_to:
        args += ["--to", a.ym_to]
    if a.skip_collect:
        args += ["--from-cache"]
    sys.argv = ["collect.py", *args]
    rc = collect.main()
    if rc:
        return rc
    sys.argv = ["build.py"]
    if (rc := build.main()):
        return rc
    sys.argv = ["export.py"]
    if (rc := export.main()):
        return rc

    dst = OUT / "gu"
    dst.mkdir(parents=True, exist_ok=True)
    for f in C.EXPORT_DIR.glob(f"gu/{SGG}*.json"):
        shutil.copy2(f, dst / f.name)
    print(f"완료 → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
