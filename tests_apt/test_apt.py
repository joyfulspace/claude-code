import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from apt_analysis import kapt, metrics_cfg as M, scorecard as S
from apt_analysis.common import Complex, anchor, idx_to_ym, monthly_trades, peak, ym_to_idx


def cx(t, p, n=None, **kw):
    return Complex(cid="1-1", sgg="11110", gu_name="종로구", name="테스트", dong="가동",
                   build_year=2015, life_trades=10, lat=37.57, lng=127.0,
                   t=[ym_to_idx(x) for x in t], p=p, n=n or [3] * len(t), amt=[0] * len(t), **kw)


class T(unittest.TestCase):
    def test_idx(self):
        self.assertEqual(idx_to_ym(ym_to_idx("2022-12")), "2022-12")

    def test_anchor_nearest_and_tie(self):
        c = cx(["2022-10", "2022-11", "2023-01"], [100, 110, 130], [1, 1, 3])
        self.assertEqual(anchor(c, M.T0), (110 * 1 + 130 * 3) / 4)      # 11월·1월 거리 1 동률 → n 가중
        self.assertIsNone(anchor(cx(["2022-08"], [1]), M.T0))             # 4개월 차이 → 없음
        self.assertEqual(anchor(cx(["2022-10", "2022-12"], [1, 2]), M.T0), 2)

    def test_peak_window(self):
        c = cx(["2020-12", "2021-01", "2022-06", "2022-07"], [999, 100, 120, 500])
        self.assertEqual(peak(c, M.PEAK_LO, M.PEAK_HI), 120)

    def test_monthly_trades(self):
        c = cx(["2025-08", "2025-09"], [1, 1], [6, 6])
        self.assertAlmostEqual(monthly_trades(c, M.T1 - 11, M.T1), 12 / 3 / 12)

    def test_percentile_ties(self):
        self.assertEqual(S.percentiles({"a": 1, "b": 1, "c": 3}), {"a": .25, "b": .25, "c": 1.0})

    def test_bins(self):
        self.assertEqual([S.bin_score(x, M.DEFENSE_BINS) for x in (0.01, -0.03, -0.08, -0.12, -0.3)],
                         [5, 4, 3, 2, 1])

    def test_parse_json_xml(self):
        j = '{"response":{"header":{"resultCode":"00"},"body":{"totalCount":1,"items":{"item":{"kaptCode":"A1","kaptdaCnt":"500"}}}}}'
        self.assertEqual(kapt.parse_response(j), ("00", 1, [{"kaptCode": "A1", "kaptdaCnt": "500"}]))
        x = "<response><header><resultCode>00</resultCode></header><body><items><item><kaptCode>A1</kaptCode><kaptdaCnt>500</kaptdaCnt></item></items><totalCount>1</totalCount></body></response>"
        self.assertEqual(kapt.parse_response(x)[2][0]["kaptdaCnt"], "500")

    def test_match(self):
        c = cx(["2022-12"], [1])
        c.name = "래미안 퍼스티지(1단지)"
        es = [{"kaptCode": "X", "kaptName": "래미안퍼스티지1단지", "as4": "가동"},
              {"kaptCode": "Y", "kaptName": "다른아파트", "as4": "가동"},
              {"kaptCode": "Z", "kaptName": "래미안퍼스티지1단지", "as4": "나동"}]
        self.assertEqual(kapt.match(c, es)["kaptCode"], "X")

    def test_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); gu = d / "gu"; gu.mkdir()
            base = ym_to_idx("2006-01")
            def add(ts, ps):
                dt, prev = [], base
                for t in ts:
                    dt.append(ym_to_idx(t) - prev); prev = ym_to_idx(t)
                return dt
            ts = ["2021-06", "2022-12", "2025-09"]
            (gu / "11110.json").write_text(json.dumps({
                "v": 1, "sgg": "11110", "name": "종로구", "base": base,
                "meta": [["좋은단지", "가동", 2020, 50, 37.57, 126.99], ["도시형생활주택A", "가동", 2020, 50, 37.57, 126.99]],
                "ids": ["11110-1", "11110-2"], "off": [0, 3, 6],
                "dt": add(ts, 0) * 2, "p": [3000, 2500, 2900] * 2, "n": [9, 9, 45] * 2, "amt": [0] * 6}))
            (gu / "11110.areas.json").write_text(json.dumps({
                "ids": ["11110-1/0", "11110-2/0"], "labels": [["84㎡", 84.0, 50], ["84㎡", 84.0, 50]]}))
            hh = d / "hh.csv"; hh.write_text("cid,kapt_code,kapt_name,households,use_date,apt_type\n"
                                              "11110-1,A,좋은단지,200,20200101,아파트\n"
                                              "11110-2,B,도시형생활주택A,200,20200101,아파트\n", encoding="utf-8")
            res = S.run([gu], d / "out", hh, d / "none.csv")
            self.assertEqual(res["passed"], 1)                            # 도시형 제외
            rows = list(__import__("csv").DictReader(open(d / "out" / "scorecard.csv", encoding="utf-8-sig")))
            self.assertAlmostEqual(float(rows[0]["recovery"]), 2900 / 2500 - 1, places=3)
            self.assertAlmostEqual(float(rows[0]["from_peak"]), 2900 / 3000 - 1, places=3)

    def test_dongtan_config_patch(self):
        hp = os.getenv("HOUSE_PJT_DIR")
        if not hp:
            self.skipTest("HOUSE_PJT_DIR 미설정")
        code = ("import sys;sys.path.insert(0,%r);import config as C;from pathlib import Path;"
                "C.DB_PATH=Path('/x/y.db');import db,build,collect,export;"
                "assert db.DB_PATH==Path('/x/y.db');print('ok')" % (hp + "/pipeline"))
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                           env={**os.environ, "DATA_GO_KR_KEY": "x"})
        self.assertIn("ok", r.stdout, r.stderr)


if __name__ == "__main__":
    unittest.main()
