"""全矩阵终检：每个 radius.csv 应为 202 行（表头+201 格），全部数值有限且 <1e6 ft。
输出 valid/bad/missing 三个清单。验收口径：valid 500, bad [], missing []。"""
import csv, glob, os
bad, have = [], set()
for f in glob.glob("out_v2/cav_*.radius.csv"):
    n = os.path.basename(f)[:7]
    try:
        rows = list(csv.reader(open(f)))
        assert len(rows) == 202, f"rows={len(rows)}"
        for r in rows[1:]:
            for x in r[1:]:
                v = float(x)
                assert v == v and abs(v) < 1e6, "NaN/inf/outlier"
        have.add(n)
    except Exception as e:
        bad.append((n, str(e)))
missing = [f"cav_{i:03d}" for i in range(500) if f"cav_{i:03d}" not in have]
print("valid:", len(have)); print("bad:", bad); print("missing:", missing)
